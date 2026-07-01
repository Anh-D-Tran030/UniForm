import hashlib
import os
import secrets
from contextlib import asynccontextmanager

import psycopg
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


# Reuse the project's existing Postgres instance / database.
DB_DSN = os.getenv("REALFORM_DSN", "postgresql://postgres:postgres@localhost:5432/realform")

# A default account is seeded on first startup so the app is usable immediately.
DEFAULT_ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
DEFAULT_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin")

PBKDF2_ITERATIONS = 200_000


def connect_db():
    return psycopg.connect(DB_DSN)


def hash_password(password, *, salt=None, iterations=PBKDF2_ITERATIONS):
    """Salted PBKDF2-SHA256. Stored as `pbkdf2_sha256$iterations$salt$hexdigest`."""
    if salt is None:
        salt = secrets.token_hex(16)
    derived = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations
    )
    return f"pbkdf2_sha256${iterations}${salt}${derived.hex()}"


def verify_password(password, stored):
    try:
        algorithm, iterations, salt, _ = stored.split("$")
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations)
    except (ValueError, AttributeError):
        return False
    candidate = hash_password(password, salt=salt, iterations=iterations)
    return secrets.compare_digest(candidate, stored)


def ensure_schema():
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS app_users (
                    id SERIAL PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    display_name TEXT,
                    is_active BOOLEAN NOT NULL DEFAULT TRUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS access_requests (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    requested_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            cur.execute("SELECT COUNT(*) FROM app_users")
            (count,) = cur.fetchone()
            if count == 0:
                cur.execute(
                    "INSERT INTO app_users (username, password_hash, display_name) "
                    "VALUES (%s, %s, %s)",
                    (
                        DEFAULT_ADMIN_USERNAME,
                        hash_password(DEFAULT_ADMIN_PASSWORD),
                        "Administrator",
                    ),
                )
        conn.commit()


@asynccontextmanager
async def lifespan(_app):
    ensure_schema()
    yield


app = FastAPI(title="UniForm Auth Service API", lifespan=lifespan)


class LoginRequest(BaseModel):
    username: str
    password: str


class AccessRequest(BaseModel):
    name: str
    email: str


@app.post("/access-requests")
def create_access_request(payload: AccessRequest):
    """Capture a beta access request from the public marketing page. This does
    NOT create a login account — an admin reviews requests and seeds accounts
    via scripts/seed_users.py."""
    name = (payload.name or "").strip()
    email = (payload.email or "").strip().lower()
    if not name or not email:
        raise HTTPException(status_code=400, detail="Name and email are required")
    local_and_domain = email.split("@")
    if len(local_and_domain) != 2 or not local_and_domain[0] or "." not in local_and_domain[1]:
        raise HTTPException(status_code=400, detail="Please enter a valid email address")

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO access_requests (name, email)
                VALUES (%s, %s)
                ON CONFLICT (email)
                DO UPDATE SET name = EXCLUDED.name, requested_at = now()
                """,
                (name, email),
            )
        conn.commit()
    return {"ok": True}


@app.post("/login")
def login(payload: LoginRequest):
    username = (payload.username or "").strip()
    if not username or not payload.password:
        raise HTTPException(status_code=400, detail="Username and password are required")

    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT password_hash, display_name, is_active FROM app_users WHERE username = %s",
                (username,),
            )
            row = cur.fetchone()

    if not row:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    password_hash, display_name, is_active = row
    if not is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    if not verify_password(payload.password, password_hash):
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return {"username": username, "display_name": display_name}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008)
