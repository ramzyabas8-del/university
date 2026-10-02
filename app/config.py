import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


def env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


DATABASE_URL = env(
    "DATABASE_URL",
    f"sqlite:///{BASE_DIR}/university_regulations.db",
)

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgres://",
        "postgresql+psycopg://",
        1,
    )
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace(
        "postgresql://",
        "postgresql+psycopg://",
        1,
    )

SECRET_KEY = env("SECRET_KEY", "dev-secret-change-me")
PYTHON_VERSION = env("PYTHON_VERSION", "3.11")

ADMIN_USERNAME = env("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = env("ADMIN_PASSWORD", "admin123")

# Gemini
GEMINI_API_KEY = env("GEMINI_API_KEY", "")
GEMINI_MODEL = env("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_EMBEDDING_MODEL = env(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

TOP_K = int(env("TOP_K", "6"))
CHUNK_SIZE = int(env("CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(env("CHUNK_OVERLAP", "250"))
