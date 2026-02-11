import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker



Base = declarative_base()

class DatabaseService:
    def __init__(self,
                 database_url):
        self.engine = create_engine(database_url, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(bind=self.engine)

    def get_or_create_user(self, db, external_user_id: str, username: str | None):
        user = db.query(User).filter(User.user_id == external_user_id).one_or_none()
        if user is None:
            user = User(user_id=external_user_id, username=username)
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            if username and user.username != username:
                user.username = username
                db.commit()
        return user


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    user_id = Column(String(128), unique=True, index=True, nullable=False)
    username = Column(String(256), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    messages = relationship("Message", back_populates="user")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    role = Column(String(32), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)
    user = relationship("User", back_populates="messages")


