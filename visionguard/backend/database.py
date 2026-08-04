"""
Database setup. Production target = PostgreSQL.
DATABASE_URL is read from the environment so the same code runs against
Postgres in deployment and SQLite locally (handy for a demo without a DB server).

Postgres example:
  export DATABASE_URL="postgresql+psycopg2://visionguard:visionguard@localhost:5432/visionguard"
SQLite example (dev only):
  export DATABASE_URL="sqlite:///./visionguard.db"
"""
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://visionguard:visionguard@localhost:5432/visionguard",
)

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
