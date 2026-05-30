from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base
from .config import DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _ensure_resolved_column():
    if not DATABASE_URL.startswith('sqlite'):
        return
    try:
        with engine.connect() as conn:
            cols = conn.execute(text("PRAGMA table_info(alerts)"))
            names = {r[1] for r in cols}
            if 'resolved' not in names:
                conn.execute(text("ALTER TABLE alerts ADD COLUMN resolved BOOLEAN DEFAULT 0"))
    except Exception:
        pass

def init_db():
    from . import models  # noqa
    Base.metadata.create_all(bind=engine)
    _ensure_resolved_column()
