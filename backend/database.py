import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey, Float, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
from sqlalchemy.dialects.postgresql import JSON # Import JSON type
from sqlalchemy.sql import func # Import func for default values

# Database setup - Use SQLite for local development if PostgreSQL not available
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./neuro.db")

# Handle SQLite vs PostgreSQL connection settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
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
    owned_shared_conversations = relationship("SharedConversation", back_populates="owner")
    shared_conversation_participations = relationship("SharedConversationParticipant", back_populates="user")
    goals = relationship("Goal", back_populates="user", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String, default="New Conversation")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation")

    # Indexes for performance
    __table_args__ = (
        Index('idx_conversation_user_updated', 'user_id', 'updated_at'),
    )

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    role = Column(String, nullable=False, index=True)
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)

    # AI Context
    cognitive_state = Column(Text, default="{}")  # JSON string

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")

    # Indexes for performance
    __table_args__ = (
        Index('idx_message_conv_timestamp', 'conversation_id', 'timestamp'),
        Index('idx_message_role_timestamp', 'role', 'timestamp'),
    )

class Memory(Base):
    __tablename__ = "memories"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    category = Column(String(50), default="general", index=True)
    importance_score = Column(Float, default=0.0, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_accessed = Column(DateTime, default=datetime.utcnow, index=True)
    accessed_count = Column(Integer, default=0)

    # Relationship
    user = relationship("User", back_populates="memories")

    # Indexes for performance
    __table_args__ = (
        Index('idx_memory_user_importance', 'user_id', 'importance_score'),
        Index('idx_memory_user_accessed', 'user_id', 'last_accessed'),
        Index('idx_memory_category_importance', 'category', 'importance_score'),
    )

class UserPreference(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    preference_type = Column(String(50), nullable=False, index=True)
    preference_value = Column(String(200), nullable=False)
    confidence_score = Column(Float, default=0.0, index=True)
    learned_from = Column(String(100), default="conversation")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Relationship
    user = relationship("User", back_populates="preferences")

    # Indexes for performance
    __table_args__ = (
        Index('idx_user_pref_type_confidence', 'user_id', 'preference_type', 'confidence_score'),
        Index('idx_user_pref_confidence', 'user_id', 'confidence_score'),
    )

class InteractionPattern(Base):
    __tablename__ = "interaction_patterns"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    pattern_type = Column(String, index=True)  # "response_time", "session_duration", etc.
    pattern_value = Column(Float)
    context_data = Column(JSON, nullable=True)  # Additional context
    detected_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    confidence_score = Column(Float, default=1.0)

    user = relationship("User", back_populates="patterns")

class SharedConversation(Base):
    __tablename__ = "shared_conversations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)
    shared_cognitive_state = Column(JSON, nullable=True)

    owner = relationship("User", back_populates="owned_shared_conversations")
    participants = relationship("SharedConversationParticipant", back_populates="shared_conversation")
    messages = relationship("SharedMessage", back_populates="shared_conversation")

class SharedConversationParticipant(Base):
    __tablename__ = "shared_conversation_participants"

    id = Column(Integer, primary_key=True, index=True)
    shared_conversation_id = Column(Integer, ForeignKey("shared_conversations.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    role = Column(String, default="participant")  # "owner", "moderator", "participant"
    joined_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    last_seen = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_typing = Column(Boolean, default=False)

    shared_conversation = relationship("SharedConversation", back_populates="participants")
    user = relationship("User", back_populates="shared_conversation_participations")

class SharedMessage(Base):
    __tablename__ = "shared_messages"

    id = Column(Integer, primary_key=True)
    shared_conversation_id = Column(Integer, ForeignKey('shared_conversations.id'))
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    role = Column(String(50), nullable=False)  # 'user' or 'assistant'
    timestamp = Column(DateTime, default=func.now())
    message_metadata = Column(JSON)  # Additional message metadata (renamed from metadata)

    shared_conversation = relationship("SharedConversation", back_populates="messages")
    user = relationship("User")

class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    shared_conversation_id = Column(Integer, ForeignKey("shared_conversations.id"), nullable=True)
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # "image", "document", "code", etc.
    file_size = Column(Integer)
    file_path = Column(String, nullable=False)
    analysis_result = Column(JSON, nullable=True)  # AI analysis of the file
    uploaded_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    user = relationship("User")
    conversation = relationship("Conversation")
    shared_conversation = relationship("SharedConversation")

class Goal(Base):
    __tablename__ = "goals"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(50), default="general", index=True)  # career, health, learning, etc.
    status = Column(String(20), default="active", index=True)  # active, completed, paused, cancelled
    priority = Column(String(10), default="medium", index=True)  # low, medium, high
    due_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_discussed_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    completion_percentage = Column(Float, default=0.0)
    
    # Relationship
    user = relationship("User", back_populates="goals")
    updates = relationship("GoalUpdate", back_populates="goal", cascade="all, delete-orphan")
    conversation_topics = relationship("ConversationTopic", back_populates="goal")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_goal_user_status', 'user_id', 'status'),
        Index('idx_goal_user_priority', 'user_id', 'priority'),
        Index('idx_goal_user_discussed', 'user_id', 'last_discussed_at'),
    )

class GoalUpdate(Base):
    __tablename__ = "goal_updates"
    
    id = Column(Integer, primary_key=True, index=True)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    update_type = Column(String(20), default="progress", index=True)  # progress, status_change, note
    content = Column(Text, nullable=False)
    previous_completion = Column(Float, default=0.0)
    new_completion = Column(Float, default=0.0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    goal = relationship("Goal", back_populates="updates")
    user = relationship("User")
    conversation = relationship("Conversation")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_goal_update_goal_date', 'goal_id', 'created_at'),
        Index('idx_goal_update_user_date', 'user_id', 'created_at'),
    )

class ConversationTopic(Base):
    __tablename__ = "conversation_topics"
    
    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False, index=True)
    goal_id = Column(Integer, ForeignKey("goals.id"), nullable=True, index=True)
    topic_name = Column(String(100), nullable=False, index=True)
    relevance_score = Column(Float, default=0.0)  # How relevant this topic is to the conversation
    mentioned_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    # Relationships
    conversation = relationship("Conversation")
    goal = relationship("Goal", back_populates="conversation_topics")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_conv_topic_conv_goal', 'conversation_id', 'goal_id'),
        Index('idx_conv_topic_topic_score', 'topic_name', 'relevance_score'),
    )

# Create tables
def create_tables():
    """Create database tables"""
    Base.metadata.create_all(bind=engine)
    print("Database tables created including UserPreference and InteractionPattern, SharedConversation, SharedConversationParticipant, SharedMessage, UploadedFile, Goal, GoalUpdate, ConversationTopic")

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()