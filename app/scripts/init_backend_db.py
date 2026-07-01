import os
from pathlib import Path

import psycopg


ROOT = Path(__file__).resolve().parents[2]
SQL_PATH = ROOT / "app" / "infra" / "sql" / "01_create_templates.sql"
DSN = os.getenv("REALFORM_DSN", "postgresql://postgres:postgres@localhost:5432/realform")


def main():
    sql = SQL_PATH.read_text(encoding="utf-8")
    with psycopg.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    print(f"Applied schema from {SQL_PATH}")


if __name__ == "__main__":
    main()
