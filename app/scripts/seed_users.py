"""Create or update a UniForm login account.

Usage (run from the repo root with the same Python env that runs AuthService):

    python scripts/seed_users.py --username alice --password "s3cret" --name "Alice Tran"
    python scripts/seed_users.py --username admin --password admin   # reset the default admin

Accounts are stored in the same Postgres `realform` database used by the app.
There is no public sign-up; accounts only exist if an admin adds them here.
"""

import argparse
import os
import sys

# Import the hashing + DB helpers from the auth service so the storage format
# always matches what the login endpoint verifies against.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from AuthService import connect_db, ensure_schema, hash_password  # noqa: E402


def upsert_user(username, password, display_name):
    ensure_schema()
    password_hash = hash_password(password)
    with connect_db() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO app_users (username, password_hash, display_name, is_active)
                VALUES (%s, %s, %s, TRUE)
                ON CONFLICT (username)
                DO UPDATE SET password_hash = EXCLUDED.password_hash,
                              display_name = EXCLUDED.display_name,
                              is_active = TRUE
                """,
                (username, password_hash, display_name),
            )
        conn.commit()


def main():
    parser = argparse.ArgumentParser(description="Create or update a UniForm login account.")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--name", default=None, help="Display name (optional)")
    args = parser.parse_args()

    upsert_user(args.username.strip(), args.password, args.name)
    print(f"User '{args.username}' is ready.")


if __name__ == "__main__":
    main()
