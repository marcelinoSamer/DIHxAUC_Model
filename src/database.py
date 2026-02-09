from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
import os

# Database URL — defaults to local SQLite, overridable via env var
# On Vercel the filesystem is read-only except /tmp
_default_db = "sqlite:///./restaurant_data.db"
if os.environ.get("VERCEL"):
    _default_db = "sqlite:////tmp/restaurant_data.db"

SQLALCHEMY_DATABASE_URL = os.environ.get("DATABASE_URL", _default_db)

# Create engine
connect_args = {"check_same_thread": False} if "sqlite" in SQLALCHEMY_DATABASE_URL else {}
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models (modern SQLAlchemy 2.0 API)
class Base(DeclarativeBase):
    pass

def get_db():
    """Dependency to get DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
