import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://localhost/neuro_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Database Models
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    # AI Personality Settings
    ai_personality = Column(Text, default="{}")  # JSON string
    preferred_mood = Column(String, default="neutral")
    energy_preference = Column(Float, default=0.5)

    # Relationships
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("Memory", back_populates="user", cascade="all, delete-orphan")
    preferences = relationship("UserPreference", back_populates="user", cascade="all, delete-orphan")
    patterns = relationship("InteractionPattern", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # AI Context
    cognitive_state = Column(Text, default="{}")  # JSON string

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    content = Column(Text, nullable=False)
    category = Column(String(50), default="general")
    importance_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow)
    accessed_count = Column(Integer, default=0)

    # Relationship
    user = relationship("User", back_populates="memories")

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    preference_type = Column(String(50), nullable=False)  # communication_style, topic_interest, response_length, etc.
    preference_value = Column(String(200), nullable=False)
    confidence_score = Column(Float, default=0.0)  # How confident we are about this preference
    learned_from = Column(String(100), default="conversation")  # conversation, explicit, feedback
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="preferences")

class InteractionPattern(Base):
    __tablename__ = "interaction_patterns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    pattern_type = Column(String(50), nullable=False)  # response_time, conversation_length, topic_switches
    pattern_data = Column(Text, nullable=False)  # JSON data
    frequency = Column(Integer, default=1)
    last_seen = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="patterns")

# Create tables
def create_tables():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created including UserPreference and InteractionPattern")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()