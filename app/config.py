import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
BASE_DIR=Path(__file__).resolve().parent.parent
def env(k,d=''): return os.getenv(k,d)
DATABASE_URL=env('DATABASE_URL',f'sqlite:///{BASE_DIR}/university_regulations.db')
if DATABASE_URL.startswith('postgres://'): DATABASE_URL=DATABASE_URL.replace('postgres://','postgresql+psycopg://',1)
elif DATABASE_URL.startswith('postgresql://'): DATABASE_URL=DATABASE_URL.replace('postgresql://','postgresql+psycopg://',1)
SECRET_KEY=env('SECRET_KEY','dev-secret-change-me')
ADMIN_USERNAME=env('ADMIN_USERNAME','admin'); ADMIN_PASSWORD=env('ADMIN_PASSWORD','admin123')
OPENAI_API_KEY=env('OPENAI_API_KEY',''); OPENAI_MODEL=env('OPENAI_MODEL','gpt-5.6-luna'); OPENAI_EMBEDDING_MODEL=env('OPENAI_EMBEDDING_MODEL','text-embedding-3-small')
TOP_K=int(env('TOP_K','6')); CHUNK_SIZE=int(env('CHUNK_SIZE','1800')); CHUNK_OVERLAP=int(env('CHUNK_OVERLAP','250'))
