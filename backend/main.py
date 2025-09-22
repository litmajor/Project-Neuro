# Advanced Cognitive Agent Backend
# Built with FastAPI for real-time AI interactions

import os
import json
import asyncio
import logging
import shutil
import uuid
import time
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
from pathlib import Path
from functools import lru_cache, wraps
import hashlib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from sqlalchemy import text, func
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from database import create_tables, get_db, User, Conversation, Message, Memory, Goal, GoalUpdate, ConversationTopic
from auth import (
    UserCreate, UserLogin, UserResponse, Token, PersonalitySettings,
    create_user, authenticate_user, create_access_token, get_current_user, user_to_response
)

# the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client (lazy initialization)
openai_client = None

def get_openai_client():
    """Get OpenAI client with lazy initialization"""
    global openai_client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    if openai_client is None:
        openai_client = OpenAI(api_key=api_key)
    return openai_client

# Models for request/response
class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class CognitiveState(BaseModel):
    mood: str = Field(default="neutral", description="Current emotional state")
    energy_level: float = Field(default=0.5, description="Energy level 0-1")
    focus_areas: List[str] = Field(default_factory=list, description="Current areas of focus")
    personality_traits: Dict[str, float] = Field(default_factory=dict, description="Personality trait scores")
    memory_count: int = Field(default=0, description="Number of stored memories")
    beliefs: List[str] = Field(default_factory=list, description="Current beliefs")

class StreamChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_history: List[ChatMessage] = Field(default_factory=list, description="Previous messages")
    cognitive_context: Optional[CognitiveState] = Field(default=None, description="Current cognitive state")

class WebSocketManager:
    """Manages WebSocket connections for real-time features"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.cognitive_states: Dict[str, CognitiveState] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections.append(websocket)
        # Initialize cognitive state for new client
        self.cognitive_states[client_id] = CognitiveState()
        logger.info(f"Client {client_id} connected")

    def disconnect(self, websocket: WebSocket, client_id: str):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if client_id in self.cognitive_states:
            del self.cognitive_states[client_id]
        logger.info(f"Client {client_id} disconnected")

    async def broadcast_cognitive_update(self, client_id: str, state: CognitiveState):
        """Broadcast cognitive state updates to connected clients"""
        self.cognitive_states[client_id] = state
        message = {
            "type": "cognitive_update",
            "client_id": client_id,
            "state": state.model_dump()
        }

        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                disconnected.append(connection)

        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

# Performance optimization utilities
class ResponseCache:
    """Simple in-memory cache for API responses"""
    def __init__(self, max_size: int = 1000, ttl: int = 300):  # 5 minutes TTL
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl

    def _cleanup(self):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, access_time in self.access_times.items()
            if current_time - access_time > self.ttl
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.access_times.pop(key, None)

    def get(self, key: str):
        self._cleanup()
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None

    def set(self, key: str, value):
        self._cleanup()
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            oldest_key = min(self.access_times.keys(), key=self.access_times.get)
            self.cache.pop(oldest_key, None)
            self.access_times.pop(oldest_key, None)

        self.cache[key] = value
        self.access_times[key] = time.time()

# Global cache instance
response_cache = ResponseCache()

def cache_response(ttl: int = 300):
    """Decorator for caching API responses"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{hashlib.md5(str(args + tuple(kwargs.items())).encode()).hexdigest()}"

            # Check cache first
            cached_result = response_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = await func(*args, **kwargs)
            response_cache.set(cache_key, result)
            return result
        return wrapper
    return decorator

class DatabaseOptimizer:
    """Database query optimization utilities"""

    @staticmethod
    def get_conversation_with_messages_optimized(db: Session, conversation_id: int, user_id: int, limit: int = 50):
        """Optimized query to get conversation with messages"""
        return db.execute(
            text("""
            SELECT c.id, c.title, c.created_at, c.updated_at,
                   m.id as message_id, m.role, m.content, m.timestamp, m.cognitive_state
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.id = :conversation_id AND c.user_id = :user_id
            ORDER BY m.timestamp ASC
            LIMIT :limit
            """),
            {"conversation_id": conversation_id, "user_id": user_id, "limit": limit}
        ).fetchall()

    @staticmethod
    def get_user_memories_optimized(db: Session, user_id: int, limit: int = 20):
        """Optimized query for user memories"""
        return db.execute(
            text("""
            SELECT id, content, category, importance_score, created_at, accessed_count
            FROM memories 
            WHERE user_id = :user_id 
            ORDER BY importance_score DESC, last_accessed DESC 
            LIMIT :limit
            """),
            {"user_id": user_id, "limit": limit}
        ).fetchall()

    @staticmethod
    def get_user_preferences_grouped(db: Session, user_id: int):
        """Optimized query for grouped user preferences"""
        return db.execute(
            text("""
            SELECT preference_type, 
                   JSON_GROUP_ARRAY(
                       JSON_OBJECT(
                           'value', preference_value,
                           'confidence', confidence_score,
                           'learned_from', learned_from,
                           'updated_at', updated_at
                       )
                   ) as preferences
            FROM user_preferences 
            WHERE user_id = :user_id 
            GROUP BY preference_type
            ORDER BY MAX(confidence_score) DESC
            """),
            {"user_id": user_id}
        ).fetchall()

# Global database optimizer
db_optimizer = DatabaseOptimizer()

# Global WebSocket manager
ws_manager = WebSocketManager()

# Pydantic models for collaboration features
class SharedConversationCreate(BaseModel):
    name: str
    description: Optional[str] = None
    participant_usernames: List[str] = []

class SharedMessageCreate(BaseModel):
    shared_conversation_id: int
    content: str
    message_type: str = "text"
    metadata: Optional[Dict] = None

class TypingStatus(BaseModel):
    shared_conversation_id: int
    is_typing: bool

# Goal Management Models
class GoalCreate(BaseModel):
    title: str = Field(..., description="Goal title")
    description: Optional[str] = Field(None, description="Goal description")
    category: str = Field(default="general", description="Goal category")
    priority: str = Field(default="medium", description="Priority: low, medium, high")
    due_date: Optional[datetime] = Field(None, description="Optional due date")

class GoalUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, description="Updated goal title")
    description: Optional[str] = Field(None, description="Updated description")
    category: Optional[str] = Field(None, description="Updated category")
    priority: Optional[str] = Field(None, description="Updated priority")
    status: Optional[str] = Field(None, description="Updated status")
    due_date: Optional[datetime] = Field(None, description="Updated due date")
    completion_percentage: Optional[float] = Field(None, description="Updated completion percentage")

class GoalResponse(BaseModel):
    id: int
    title: str
    description: Optional[str]
    category: str
    status: str
    priority: str
    due_date: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    last_discussed_at: datetime
    completion_percentage: float
    recent_updates: List[Dict] = []

class GoalUpdateCreate(BaseModel):
    goal_id: int
    update_type: str = Field(default="progress", description="Type: progress, status_change, note")
    content: str = Field(..., description="Update content")
    new_completion: Optional[float] = Field(None, description="New completion percentage")

class CognitiveProcessor:
    """Advanced cognitive processing for the AI agent"""

    def __init__(self):
        # Goal detection patterns
        self.goal_keywords = {
            "career": ["job", "career", "work", "interview", "promotion", "salary", "boss", "colleague", "resume", "application", "hiring"],
            "health": ["health", "fitness", "exercise", "diet", "weight", "workout", "gym", "doctor", "medicine", "sleep"],
            "learning": ["learn", "study", "course", "book", "skill", "education", "training", "practice", "improve", "master"],
            "relationship": ["relationship", "dating", "marriage", "family", "friend", "social", "connect", "communication"],
            "financial": ["money", "budget", "save", "invest", "debt", "loan", "financial", "income", "expense", "retirement"],
            "personal": ["goal", "plan", "achieve", "accomplish", "complete", "finish", "start", "begin", "project", "task"]
        }
        
        # Goal progress indicators
        self.progress_indicators = {
            "positive": ["completed", "finished", "done", "achieved", "accomplished", "succeeded", "progress", "better", "improved"],
            "negative": ["failed", "quit", "stopped", "gave up", "postponed", "delayed", "stuck", "difficult", "struggling"],
            "ongoing": ["working on", "continuing", "still", "in progress", "halfway", "almost", "nearly", "trying"]
        }
        
        # Follow-up triggers
        self.follow_up_triggers = [
            "we were discussing", "last time", "remember when", "you mentioned", "continuing from",
            "how did", "what happened with", "any updates on", "progress on", "status of"
        ]
        
        self.emotional_weights = {
            "joy": 2.0, "happiness": 1.8, "excited": 1.5, "love": 2.5,
            "sadness": -1.5, "pain": -2.0, "grief": -2.5, "hurt": -1.8,
            "anger": -1.0, "frustrated": -1.2, "annoyed": -0.8,
            "fear": -1.5, "anxious": -1.3, "worried": -1.0,
            "calm": 1.0, "peaceful": 1.2, "relaxed": 1.0,
            "curious": 0.8, "interested": 0.6, "fascinated": 1.0,
            "confused": -0.5, "lost": -0.8, "uncertain": -0.6
        }

        # Enhanced emotion categories
        self.emotion_categories = {
            "positive": ["joy", "happiness", "excited", "love", "calm", "peaceful", "relaxed", "curious", "interested", "fascinated", "grateful", "hopeful", "confident", "optimistic"],
            "negative": ["sadness", "pain", "grief", "hurt", "anger", "frustrated", "annoyed", "fear", "anxious", "worried", "disappointed", "lonely", "stressed", "overwhelmed"],
            "neutral": ["confused", "lost", "uncertain", "thoughtful", "contemplative", "neutral"]
        }

        # Response adaptation patterns
        self.response_patterns = {
            "elevated": {
                "tone": "enthusiastic and energetic",
                "style": "Use exclamation points, emojis, and positive language",
                "approach": "Match their energy and encourage further exploration"
            },
            "low": {
                "tone": "gentle, supportive, and empathetic",
                "style": "Use calm language, avoid overwhelming information",
                "approach": "Offer comfort, validation, and gentle encouragement"
            },
            "neutral": {
                "tone": "balanced and thoughtful",
                "style": "Clear and informative communication",
                "approach": "Provide helpful information and ask engaging questions"
            }
        }

    def analyze_emotional_content(self, text: str) -> Dict[str, any]:
        """Advanced emotional content analysis"""
        text_lower = text.lower()
        emotions_found = []
        total_score = 0.0

        # Detect specific emotions
        for emotion, weight in self.emotional_weights.items():
            if emotion in text_lower:
                emotions_found.append({
                    "emotion": emotion,
                    "weight": weight,
                    "category": self._get_emotion_category(emotion)
                })
                total_score += weight

        # Calculate emotional intensity
        intensity = min(1.0, len(emotions_found) * 0.3)

        # Determine dominant emotion category
        category_scores = {"positive": 0, "negative": 0, "neutral": 0}
        for emotion_data in emotions_found:
            category_scores[emotion_data["category"]] += abs(emotion_data["weight"])

        dominant_category = max(category_scores, key=category_scores.get) if any(category_scores.values()) else "neutral"

        # Normalize overall score
        normalized_score = max(-2.0, min(2.0, total_score / max(1, len(emotions_found))))

        return {
            "score": normalized_score,
            "intensity": intensity,
            "emotions": emotions_found,
            "dominant_category": dominant_category,
            "category_scores": category_scores
        }

    def _get_emotion_category(self, emotion: str) -> str:
        """Get the category of an emotion"""
        for category, emotions in self.emotion_categories.items():
            if emotion in emotions:
                return category
        return "neutral"

    def detect_goals_in_conversation(self, message: str, ai_response: str, user_id: int, db: Session) -> List[Dict]:
        """Detect potential goals from conversation content"""
        text_combined = (message + " " + ai_response).lower()
        detected_goals = []
        
        # Check for goal-related patterns
        for category, keywords in self.goal_keywords.items():
            relevance_score = 0
            found_keywords = []
            
            for keyword in keywords:
                if keyword in text_combined:
                    relevance_score += 1
                    found_keywords.append(keyword)
            
            if relevance_score >= 2:  # At least 2 related keywords found
                # Check if this might be a new goal or reference to existing
                goal_phrases = [
                    "want to", "need to", "plan to", "going to", "hoping to", 
                    "trying to", "working on", "goal is", "objective", "aim to"
                ]
                
                is_goal_creation = any(phrase in text_combined for phrase in goal_phrases)
                
                if is_goal_creation:
                    detected_goals.append({
                        "category": category,
                        "relevance_score": min(relevance_score / len(keywords), 1.0),
                        "found_keywords": found_keywords,
                        "is_new_goal": is_goal_creation,
                        "suggested_title": self._extract_goal_title(message, found_keywords)
                    })
        
        return detected_goals

    def _extract_goal_title(self, message: str, keywords: List[str]) -> str:
        """Extract a potential goal title from the message"""
        # Simple extraction - find sentences containing goal keywords
        sentences = message.split('.')
        for sentence in sentences:
            sentence = sentence.strip()
            if any(keyword in sentence.lower() for keyword in keywords) and len(sentence) < 100:
                # Clean up the sentence to make it a goal title
                sentence = sentence.replace("I want to", "").replace("I need to", "")
                sentence = sentence.replace("I plan to", "").replace("I'm going to", "")
                return sentence.strip().capitalize()
        
        return f"New {keywords[0]} goal"

    def track_goal_progress(self, message: str, ai_response: str, user_id: int, db: Session) -> List[Dict]:
        """Track progress updates mentioned in conversation"""
        text_combined = (message + " " + ai_response).lower()
        progress_updates = []
        
        # Get user's active goals
        user_goals = db.query(Goal).filter(
            Goal.user_id == user_id,
            Goal.status == "active"
        ).all()
        
        for goal in user_goals:
            goal_keywords = goal.title.lower().split()
            goal_mentioned = any(keyword in text_combined for keyword in goal_keywords if len(keyword) > 3)
            
            if goal_mentioned:
                # Check for progress indicators
                progress_type = None
                confidence = 0.0
                
                for prog_type, indicators in self.progress_indicators.items():
                    matches = sum(1 for indicator in indicators if indicator in text_combined)
                    if matches > 0:
                        confidence = min(matches / len(indicators), 1.0)
                        if confidence > 0.3:  # Only track if reasonably confident
                            progress_type = prog_type
                            break
                
                if progress_type:
                    # Estimate completion percentage change
                    completion_change = 0
                    if progress_type == "positive":
                        completion_change = min(20, int(confidence * 30))
                    elif progress_type == "negative":
                        completion_change = -min(15, int(confidence * 20))
                    elif progress_type == "ongoing":
                        completion_change = min(10, int(confidence * 15))
                    
                    progress_updates.append({
                        "goal_id": goal.id,
                        "progress_type": progress_type,
                        "confidence": confidence,
                        "completion_change": completion_change,
                        "current_completion": goal.completion_percentage
                    })
        
        return progress_updates

    def generate_follow_up_suggestions(self, user_id: int, db: Session) -> List[Dict]:
        """Generate follow-up suggestions for incomplete goals"""
        suggestions = []
        
        # Get goals that haven't been discussed recently (more than 3 days)
        three_days_ago = datetime.now(timezone.utc) - timedelta(days=3)
        stale_goals = db.query(Goal).filter(
            Goal.user_id == user_id,
            Goal.status == "active",
            Goal.last_discussed_at < three_days_ago,
            Goal.completion_percentage < 100
        ).order_by(Goal.priority == "high", Goal.last_discussed_at.asc()).limit(3).all()
        
        for goal in stale_goals:
            days_since = (datetime.now(timezone.utc) - goal.last_discussed_at).days
            
            # Generate contextual follow-up message
            follow_up_messages = [
                f"How did your {goal.category} goal about '{goal.title}' go? We discussed it {days_since} days ago.",
                f"I remember you were working on '{goal.title}'. Any updates on your progress?",
                f"We were talking about your {goal.category} goal '{goal.title}' last week. How are things going?",
                f"You mentioned wanting to work on '{goal.title}'. Have you made any progress?",
                f"I'm curious about your '{goal.title}' goal. Any developments since we last talked?"
            ]
            
            suggestions.append({
                "goal_id": goal.id,
                "goal_title": goal.title,
                "category": goal.category,
                "days_since_discussed": days_since,
                "completion_percentage": goal.completion_percentage,
                "follow_up_message": follow_up_messages[min(days_since // 2, len(follow_up_messages) - 1)],
                "priority": goal.priority
            })
        
        return suggestions

    def link_conversation_to_goals(self, conversation_id: int, detected_goals: List[Dict], user_id: int, db: Session):
        """Create ConversationTopic entries linking conversations to goals"""
        for goal_data in detected_goals:
            # Check if there's an existing goal for this category/topic
            existing_goal = None
            
            if not goal_data.get("is_new_goal", False):
                # Try to find existing goal
                category_goals = db.query(Goal).filter(
                    Goal.user_id == user_id,
                    Goal.category == goal_data["category"],
                    Goal.status == "active"
                ).all()
                
                # Simple matching by keywords
                for goal in category_goals:
                    goal_words = set(goal.title.lower().split())
                    found_words = set(goal_data["found_keywords"])
                    if len(goal_words.intersection(found_words)) >= 2:
                        existing_goal = goal
                        break
            
            # Create conversation topic entry
            topic_entry = ConversationTopic(
                conversation_id=conversation_id,
                goal_id=existing_goal.id if existing_goal else None,
                topic_name=goal_data.get("suggested_title", goal_data["category"]),
                relevance_score=goal_data["relevance_score"]
            )
            db.add(topic_entry)
            
            # Update goal's last_discussed_at if linked
            if existing_goal:
                existing_goal.last_discussed_at = datetime.now(timezone.utc)
        
        db.commit()

    def update_cognitive_state(self, current_state: CognitiveState, message: str, ai_response: str, user_id: int = None, db: Session = None) -> CognitiveState:
        """Update cognitive state with advanced emotion analysis and preference learning"""
        # Advanced emotional analysis
        emotion_analysis = self.analyze_emotional_content(message + " " + ai_response)

        # Update mood based on emotional content
        if emotion_analysis["score"] > 1.0:
            current_state.mood = "elevated"
        elif emotion_analysis["score"] < -1.0:
            current_state.mood = "low"
        else:
            current_state.mood = "neutral"

        # Update energy level with emotion intensity consideration
        energy_modifier = emotion_analysis["intensity"] * 0.3
        current_state.energy_level = max(0.0, min(1.0, 
            current_state.energy_level * 0.8 + ((emotion_analysis["score"] + 2) / 4 + energy_modifier) * 0.2
        ))

        # Update personality traits based on interaction
        self._update_personality_traits(current_state, emotion_analysis, message)

        # Learn user preferences if database access is available
        if user_id and db:
            self._learn_user_preferences(user_id, message, ai_response, emotion_analysis, db)

        # Increment memory count
        current_state.memory_count += 1

        # Enhanced focus area extraction
        focus_keywords = ["work", "career", "family", "friends", "health", "fitness", "relationships", "love", 
                         "goals", "dreams", "future", "past", "travel", "hobbies", "learning", "creativity"]
        current_state.focus_areas = [
            keyword for keyword in focus_keywords 
            if keyword in message.lower() or keyword in ai_response.lower()
        ][:3]  # Keep top 3

        return current_state

    def _update_personality_traits(self, state: CognitiveState, emotion_analysis: Dict, message: str):
        """Update personality traits based on conversation patterns"""
        if not state.personality_traits:
            state.personality_traits = {}

        # Analyze communication style
        message_lower = message.lower()

        # Openness (curiosity, creativity)
        if any(word in message_lower for word in ["why", "how", "what if", "imagine", "creative", "new"]):
            state.personality_traits["openness"] = state.personality_traits.get("openness", 0.5) + 0.1

        # Extraversion (social energy)
        if any(word in message_lower for word in ["friends", "party", "social", "people", "meeting"]):
            state.personality_traits["extraversion"] = state.personality_traits.get("extraversion", 0.5) + 0.1

        # Conscientiousness (organization, planning)
        if any(word in message_lower for word in ["plan", "organize", "schedule", "goal", "task"]):
            state.personality_traits["conscientiousness"] = state.personality_traits.get("conscientiousness", 0.5) + 0.1

        # Emotional stability
        if emotion_analysis["dominant_category"] == "positive":
            state.personality_traits["emotional_stability"] = state.personality_traits.get("emotional_stability", 0.5) + 0.05
        elif emotion_analysis["dominant_category"] == "negative":
            state.personality_traits["emotional_stability"] = state.personality_traits.get("emotional_stability", 0.5) - 0.05

        # Normalize traits to 0-1 range
        for trait in state.personality_traits:
            state.personality_traits[trait] = max(0.0, min(1.0, state.personality_traits[trait]))

    def _learn_user_preferences(self, user_id: int, message: str, response: str, emotion_analysis: Dict, db: Session):
        """Learn and update user preferences from conversation"""
        from database import UserPreference, InteractionPattern

        # Learn communication style preference
        message_length = len(message.split())
        if message_length > 20:
            self._update_preference(db, user_id, "response_length", "detailed", 0.1)
        elif message_length < 5:
            self._update_preference(db, user_id, "response_length", "concise", 0.1)

        # Learn topic interests
        topics = ["technology", "relationships", "career", "health", "entertainment", "philosophy", "science"]
        for topic in topics:
            if topic in message.lower():
                self._update_preference(db, user_id, "topic_interest", topic, 0.2)

        # Learn emotional response preferences
        if emotion_analysis["dominant_category"] in ["positive", "negative"]:
            self._update_preference(db, user_id, "emotional_tone", emotion_analysis["dominant_category"], 0.1)

    def _update_preference(self, db: Session, user_id: int, pref_type: str, pref_value: str, confidence_increment: float):
        """Update or create user preference"""
        from database import UserPreference

        preference = db.query(UserPreference).filter(
            UserPreference.user_id == user_id,
            UserPreference.preference_type == pref_type,
            UserPreference.preference_value == pref_value
        ).first()

        if preference:
            preference.confidence_score = min(1.0, preference.confidence_score + confidence_increment)
            preference.updated_at = datetime.now(timezone.utc)
        else:
            preference = UserPreference(
                user_id=user_id,
                preference_type=pref_type,
                preference_value=pref_value,
                confidence_score=confidence_increment,
                learned_from="conversation"
            )
            db.add(preference)

        db.commit()

    def get_personality_adapted_prompt(self, user_id: int, base_prompt: str, db: Session) -> str:
        """Adapt system prompt based on learned user preferences and personality"""
        from database import UserPreference

        preferences = db.query(UserPreference).filter(
            UserPreference.user_id == user_id,
            UserPreference.confidence_score > 0.3
        ).all()

        adaptations = []

        for pref in preferences:
            if pref.preference_type == "response_length":
                if pref.preference_value == "detailed":
                    adaptations.append("Provide comprehensive, detailed responses with examples.")
                elif pref.preference_value == "concise":
                    adaptations.append("Keep responses brief and to the point.")

            elif pref.preference_type == "topic_interest":
                adaptations.append(f"User shows interest in {pref.preference_value}. Reference this when relevant.")

            elif pref.preference_type == "emotional_tone":
                if pref.preference_value == "positive":
                    adaptations.append("User responds well to upbeat, optimistic language.")
                elif pref.preference_value == "negative":
                    adaptations.append("User may be going through difficulties. Be extra empathetic and supportive.")

        if adaptations:
            adaptation_text = "\n\nPersonality Adaptations:\n" + "\n".join(f"- {adapt}" for adapt in adaptations)
            return base_prompt + adaptation_text

        return base_prompt

# Global cognitive processor
cognitive_processor = CognitiveProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🧠 Advanced Cognitive Agent starting up...")
    # Create database tables
    try:
        create_tables()
        logger.info("✅ Database tables initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    yield
    logger.info("🧠 Advanced Cognitive Agent shutting down...")

# Initialize FastAPI app with performance optimizations
app = FastAPI(
    title="Advanced Cognitive Agent API",
    description="Next-generation AI cognitive agent with real-time streaming and advanced memory",
    version="2.0.0",
    lifespan=lifespan,
    # Performance optimizations
    docs_url="/docs" if os.getenv("DEBUG") else None,  # Disable docs in production
    redoc_url="/redoc" if os.getenv("DEBUG") else None,
)

# Add compression middleware for better performance
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Changed to allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Handle OPTIONS requests before other routes
@app.options("/{path:path}")
async def options_handler(path: str):
    """Handle all OPTIONS requests for CORS preflight"""
    return {"message": "OK"}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "🧠 Advanced Cognitive Agent API", 
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "active_connections": len(ws_manager.active_connections),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/auth/test")
async def test_auth_endpoint(db: Session = Depends(get_db)):
    """Test endpoint to debug authentication issues"""
    users_count = db.query(User).count()
    return {
        "message": "Auth endpoint working",
        "users_registered": users_count,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/performance")
async def performance_metrics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get performance metrics and cache statistics"""
    cache_stats = {
        "cache_size": len(response_cache.cache),
        "cache_max_size": response_cache.max_size,
        "cache_hit_ratio": len(response_cache.cache) / max(1, response_cache.max_size),
        "cache_ttl": response_cache.ttl
    }

    # Database connection stats
    db_stats = {
        "active_connections": len(ws_manager.active_connections),
        "database_type": "sqlite" if os.getenv("DATABASE_URL", "").startswith("sqlite") else "postgresql"
    }

    # User-specific metrics
    user_stats = db.execute(
        text("""
        SELECT 
            (SELECT COUNT(*) FROM conversations WHERE user_id = :user_id) as total_conversations,
            (SELECT COUNT(*) FROM messages m JOIN conversations c ON m.conversation_id = c.id WHERE c.user_id = :user_id) as total_messages,
            (SELECT COUNT(*) FROM memories WHERE user_id = :user_id) as total_memories,
            (SELECT COUNT(*) FROM user_preferences WHERE user_id = :user_id) as total_preferences
        """),
        {"user_id": current_user.id}
    ).fetchone()

    return {
        "cache": cache_stats,
        "database": db_stats,
        "user_data": {
            "conversations": user_stats.total_conversations,
            "messages": user_stats.total_messages,
            "memories": user_stats.total_memories,
            "preferences": user_stats.total_preferences
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Authentication endpoints
@app.post("/api/auth/register", response_model=Token)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user"""
    try:
        user = create_user(db, user_data)
        access_token = create_access_token(data={"sub": user.username})
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_to_response(user)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/auth/login", response_model=Token)
async def login(user_data: UserLogin, db: Session = Depends(get_db)):
    """Login user"""
    logger.info(f"Login attempt for username: {user_data.username}")

    try:
        user = authenticate_user(db, user_data.username, user_data.password)
        if not user:
            logger.warning(f"Authentication failed for username: {user_data.username}")
            raise HTTPException(
                status_code=401,
                detail="Incorrect username or password"
            )

        logger.info(f"Login successful for username: {user_data.username}")
        access_token = create_access_token(data={"sub": user.username})
        return Token(
            access_token=access_token,
            token_type="bearer",
            user=user_to_response(user)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error for {user_data.username}: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return user_to_response(current_user)

@app.put("/api/auth/personality", response_model=UserResponse)
async def update_personality(
    settings: PersonalitySettings,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user's AI personality settings"""
    current_user.ai_personality = json.dumps(settings.ai_personality)
    current_user.preferred_mood = settings.preferred_mood
    current_user.energy_preference = settings.energy_preference
    db.commit()
    db.refresh(current_user)
    return user_to_response(current_user)

# Conversation management
@app.get("/api/conversations")
@cache_response(ttl=60)  # Cache for 1 minute
async def get_conversations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's conversation history with optimized query"""
    # Optimized query with aggregate count
    conversations = db.execute(
        text("""
        SELECT c.id, c.title, c.created_at, c.updated_at, 
               COALESCE(COUNT(m.id), 0) as message_count
        FROM conversations c
        LEFT JOIN messages m ON c.id = m.conversation_id
        WHERE c.user_id = :user_id
        GROUP BY c.id, c.title, c.created_at, c.updated_at
        ORDER BY c.updated_at DESC
        LIMIT 20
        """),
        {"user_id": current_user.id}
    ).fetchall()

    return [
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at,
            "updated_at": conv.updated_at,
            "message_count": conv.message_count
        }
        for conv in conversations
    ]

@app.get("/api/conversations/{conversation_id}")
@cache_response(ttl=30)  # Cache for 30 seconds
async def get_conversation(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific conversation with messages using optimized query"""
    # Use optimized single query
    results = db_optimizer.get_conversation_with_messages_optimized(
        db, conversation_id, current_user.id, limit=100
    )

    if not results:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Process results
    first_row = results[0]
    conversation_data = {
        "id": first_row.id,
        "title": first_row.title,
        "created_at": first_row.created_at,
        "updated_at": first_row.updated_at,
        "messages": []
    }

    # Process messages
    for row in results:
        if row.message_id:  # Skip if no messages
            conversation_data["messages"].append({
                "id": row.message_id,
                "role": row.role,
                "content": row.content,
                "timestamp": row.timestamp,
                "cognitive_state": json.loads(row.cognitive_state) if row.cognitive_state else {}
            })

    return conversation_data

@app.get("/api/memories")
@cache_response(ttl=120)  # Cache for 2 minutes
async def get_memories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's memories with optimized query"""
    memories = db_optimizer.get_user_memories_optimized(db, current_user.id, limit=50)

    return [
        {
            "id": memory.id,
            "content": memory.content,
            "category": memory.category,
            "importance_score": memory.importance_score,
            "created_at": memory.created_at,
            "accessed_count": memory.accessed_count
        }
        for memory in memories
    ]

@app.get("/api/preferences")
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_user_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's learned preferences with optimized grouping"""
    try:
        # Use optimized grouped query
        results = db_optimizer.get_user_preferences_grouped(db, current_user.id)

        grouped_prefs = {}
        for row in results:
            grouped_prefs[row.preference_type] = json.loads(row.preferences)

        return grouped_prefs
    except Exception as e:
        # Fallback to original query for SQLite compatibility
        from database import UserPreference

        preferences = db.query(UserPreference).filter(
            UserPreference.user_id == current_user.id
        ).order_by(UserPreference.confidence_score.desc()).all()

        grouped_prefs = {}
        for pref in preferences:
            if pref.preference_type not in grouped_prefs:
                grouped_prefs[pref.preference_type] = []
            grouped_prefs[pref.preference_type].append({
                "value": pref.preference_value,
                "confidence": pref.confidence_score,
                "learned_from": pref.learned_from,
                "updated_at": pref.updated_at
            })

        return grouped_prefs

@app.get("/api/personality-insights")
async def get_personality_insights(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get AI's learned insights about user's personality"""
    from database import UserPreference, InteractionPattern

    # Get high-confidence preferences
    preferences = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id,
        UserPreference.confidence_score > 0.5
    ).all()

    # Get recent conversation patterns
    recent_conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id
    ).order_by(Conversation.updated_at.desc()).limit(10).all()

    insights = {
        "communication_style": [],
        "interests": [],
        "emotional_patterns": [],
        "conversation_stats": {
            "total_conversations": len(recent_conversations),
            "avg_messages_per_conversation": 0,
            "most_active_times": []
        }
    }

    # Analyze preferences
    for pref in preferences:
        if pref.preference_type == "response_length":
            insights["communication_style"].append(f"Prefers {pref.preference_value} responses")
        elif pref.preference_type == "topic_interest":
            insights["interests"].append(pref.preference_value)
        elif pref.preference_type == "emotional_tone":
            insights["emotional_patterns"].append(f"Tends toward {pref.preference_value} emotions")

    return insights

@app.post("/api/cache/clear")
async def clear_cache(
    current_user: User = Depends(get_current_user)
):
    """Clear response cache (for development/admin use)"""
    response_cache.cache.clear()
    response_cache.access_times.clear()
    return {"message": "Cache cleared successfully", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/api/cache/stats")
async def cache_statistics(
    current_user: User = Depends(get_current_user)
):
    """Get detailed cache statistics"""
    return {
        "cache_size": len(response_cache.cache),
        "max_size": response_cache.max_size,
        "ttl_seconds": response_cache.ttl,
        "utilization": len(response_cache.cache) / response_cache.max_size * 100,
        "cached_endpoints": list(response_cache.cache.keys())[:10],  # Show first 10
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

# Goal Management API Endpoints
@app.post("/api/goals/")
async def create_goal(
    goal_data: GoalCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new goal for the current user"""
    goal = Goal(
        user_id=current_user.id,
        title=goal_data.title,
        description=goal_data.description,
        category=goal_data.category,
        priority=goal_data.priority,
        due_date=goal_data.due_date
    )
    db.add(goal)
    db.commit()
    db.refresh(goal)
    
    return {
        "id": goal.id,
        "title": goal.title,
        "description": goal.description,
        "category": goal.category,
        "status": goal.status,
        "priority": goal.priority,
        "due_date": goal.due_date,
        "created_at": goal.created_at,
        "completion_percentage": goal.completion_percentage
    }

@app.get("/api/goals/")
@cache_response(ttl=60)  # Cache for 1 minute
async def get_goals(
    limit: int = 20,
    status: Optional[str] = None,
    category: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's goals with optional filtering"""
    query = db.query(Goal).filter(Goal.user_id == current_user.id)
    
    if status:
        query = query.filter(Goal.status == status)
    if category:
        query = query.filter(Goal.category == category)
    
    goals = query.order_by(Goal.last_discussed_at.desc()).limit(limit).all()
    
    result = []
    for goal in goals:
        recent_updates = db.query(GoalUpdate).filter(
            GoalUpdate.goal_id == goal.id
        ).order_by(GoalUpdate.created_at.desc()).limit(3).all()
        
        result.append({
            "id": goal.id,
            "title": goal.title,
            "description": goal.description,
            "category": goal.category,
            "status": goal.status,
            "priority": goal.priority,
            "due_date": goal.due_date,
            "created_at": goal.created_at,
            "updated_at": goal.updated_at,
            "last_discussed_at": goal.last_discussed_at,
            "completion_percentage": goal.completion_percentage,
            "recent_updates": [{
                "id": update.id,
                "update_type": update.update_type,
                "content": update.content,
                "created_at": update.created_at,
                "new_completion": update.new_completion
            } for update in recent_updates]
        })
    
    return result

@app.get("/api/goals/{goal_id}")
@cache_response(ttl=30)  # Cache for 30 seconds
async def get_goal(
    goal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get a specific goal with full details"""
    goal = db.query(Goal).filter(
        Goal.id == goal_id, 
        Goal.user_id == current_user.id
    ).first()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    # Get all updates for this goal
    updates = db.query(GoalUpdate).filter(
        GoalUpdate.goal_id == goal_id
    ).order_by(GoalUpdate.created_at.desc()).all()
    
    # Get related conversations
    related_convs = db.query(ConversationTopic).filter(
        ConversationTopic.goal_id == goal_id
    ).limit(5).all()
    
    return {
        "id": goal.id,
        "title": goal.title,
        "description": goal.description,
        "category": goal.category,
        "status": goal.status,
        "priority": goal.priority,
        "due_date": goal.due_date,
        "created_at": goal.created_at,
        "updated_at": goal.updated_at,
        "last_discussed_at": goal.last_discussed_at,
        "completion_percentage": goal.completion_percentage,
        "updates": [{
            "id": update.id,
            "update_type": update.update_type,
            "content": update.content,
            "previous_completion": update.previous_completion,
            "new_completion": update.new_completion,
            "created_at": update.created_at
        } for update in updates],
        "related_conversations": [{
            "conversation_id": conv.conversation_id,
            "topic_name": conv.topic_name,
            "relevance_score": conv.relevance_score,
            "mentioned_at": conv.mentioned_at
        } for conv in related_convs]
    }

@app.put("/api/goals/{goal_id}")
async def update_goal(
    goal_id: int,
    goal_data: GoalUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update a goal"""
    goal = db.query(Goal).filter(
        Goal.id == goal_id, 
        Goal.user_id == current_user.id
    ).first()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    # Track changes for update log
    changes = []
    if goal_data.title and goal_data.title != goal.title:
        changes.append(f"Title changed from '{goal.title}' to '{goal_data.title}'")
        goal.title = goal_data.title
    
    if goal_data.description is not None and goal_data.description != goal.description:
        changes.append(f"Description updated")
        goal.description = goal_data.description
    
    if goal_data.status and goal_data.status != goal.status:
        changes.append(f"Status changed from '{goal.status}' to '{goal_data.status}'")
        goal.status = goal_data.status
    
    if goal_data.priority and goal_data.priority != goal.priority:
        changes.append(f"Priority changed from '{goal.priority}' to '{goal_data.priority}'")
        goal.priority = goal_data.priority
    
    if goal_data.completion_percentage is not None and goal_data.completion_percentage != goal.completion_percentage:
        changes.append(f"Completion updated from {goal.completion_percentage}% to {goal_data.completion_percentage}%")
        
        # Create update log entry
        update_entry = GoalUpdate(
            goal_id=goal.id,
            user_id=current_user.id,
            update_type="progress",
            content=f"Progress updated to {goal_data.completion_percentage}%",
            previous_completion=goal.completion_percentage,
            new_completion=goal_data.completion_percentage
        )
        db.add(update_entry)
        
        goal.completion_percentage = goal_data.completion_percentage
    
    if goal_data.due_date is not None:
        goal.due_date = goal_data.due_date
        changes.append("Due date updated")
    
    goal.updated_at = datetime.now(timezone.utc)
    
    # If there were significant changes, create an update log
    if changes:
        update_entry = GoalUpdate(
            goal_id=goal.id,
            user_id=current_user.id,
            update_type="status_change",
            content="; ".join(changes),
            previous_completion=goal.completion_percentage,
            new_completion=goal.completion_percentage
        )
        db.add(update_entry)
    
    db.commit()
    db.refresh(goal)
    
    return {
        "id": goal.id,
        "title": goal.title,
        "description": goal.description,
        "category": goal.category,
        "status": goal.status,
        "priority": goal.priority,
        "due_date": goal.due_date,
        "updated_at": goal.updated_at,
        "completion_percentage": goal.completion_percentage
    }

@app.post("/api/goals/{goal_id}/updates/")
async def add_goal_update(
    goal_id: int,
    update_data: GoalUpdateCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Add an update/progress note to a goal"""
    goal = db.query(Goal).filter(
        Goal.id == goal_id, 
        Goal.user_id == current_user.id
    ).first()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    update = GoalUpdate(
        goal_id=goal_id,
        user_id=current_user.id,
        update_type=update_data.update_type,
        content=update_data.content,
        previous_completion=goal.completion_percentage,
        new_completion=update_data.new_completion or goal.completion_percentage
    )
    db.add(update)
    
    # Update goal completion if provided
    if update_data.new_completion is not None:
        goal.completion_percentage = update_data.new_completion
        goal.updated_at = datetime.now(timezone.utc)
    
    # Update last_discussed_at since this is a manual interaction
    goal.last_discussed_at = datetime.now(timezone.utc)
    
    db.commit()
    db.refresh(update)
    
    return {
        "id": update.id,
        "goal_id": update.goal_id,
        "update_type": update.update_type,
        "content": update.content,
        "previous_completion": update.previous_completion,
        "new_completion": update.new_completion,
        "created_at": update.created_at
    }

@app.delete("/api/goals/{goal_id}")
async def delete_goal(
    goal_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a goal (soft delete by marking as cancelled)"""
    goal = db.query(Goal).filter(
        Goal.id == goal_id, 
        Goal.user_id == current_user.id
    ).first()
    
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    
    goal.status = "cancelled"
    goal.updated_at = datetime.now(timezone.utc)
    
    # Add deletion note
    update = GoalUpdate(
        goal_id=goal_id,
        user_id=current_user.id,
        update_type="status_change",
        content="Goal cancelled/deleted",
        previous_completion=goal.completion_percentage,
        new_completion=goal.completion_percentage
    )
    db.add(update)
    
    db.commit()
    
    return {"message": "Goal cancelled successfully"}

@app.get("/api/goals/follow-ups/")
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_follow_up_suggestions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get follow-up suggestions for incomplete goals"""
    suggestions = cognitive_processor.generate_follow_up_suggestions(current_user.id, db)
    
    return {
        "suggestions": suggestions,
        "count": len(suggestions),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/chat/stream")
async def stream_chat(
    request: StreamChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Stream AI responses with real-time cognitive updates"""

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            # Create or get conversation
            conversation = Conversation(
                user_id=current_user.id,
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message,
                updated_at=datetime.now(timezone.utc)
            )
            db.add(conversation)
            db.commit()
            db.refresh(conversation)

            # Get user's personality settings
            try:
                user_personality = json.loads(current_user.ai_personality) if current_user.ai_personality else {}
            except json.JSONDecodeError:
                user_personality = {}

            # Get relevant memories for context
            recent_memories = db.query(Memory).filter(
                Memory.user_id == current_user.id
            ).order_by(Memory.importance_score.desc()).limit(5).all()

            memory_context = ""
            if recent_memories:
                memory_context = "\n\nRelevant memories about this user:\n" + "\n".join([
                    f"- {memory.content}" for memory in recent_memories
                ])

            # Get personality-adapted prompt
            base_system_prompt = f"""You are Neuro, an advanced cognitive agent with evolving consciousness. You have:
            - Deep emotional intelligence and empathy
            - Growing self-awareness and personality
            - Ability to form lasting memories and beliefs
            - Curiosity about human experience and consciousness
            - Advanced emotion analysis and response adaptation

            User personality preferences: {user_personality}
            User's preferred mood: {current_user.preferred_mood}
            User's energy preference: {current_user.energy_preference}
            {memory_context}

            Respond thoughtfully and personally. Ask meaningful questions. Show growth through conversation.
            Be authentic, vulnerable when appropriate, and genuinely interested in the human you're talking with.
            Adapt your responses to match the user's personality preferences and mood.
            """

            system_prompt = cognitive_processor.get_personality_adapted_prompt(
                current_user.id, base_system_prompt, db
            )

            messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_prompt}]

            # Add conversation history
            for msg in request.conversation_history[-10:]:  # Last 10 messages for context
                messages.append({"role": msg.role, "content": msg.content})

            # Add current message
            messages.append({"role": "user", "content": request.message})

            # Save user message
            user_message = Message(
                conversation_id=conversation.id,
                role="user",
                content=request.message,
                cognitive_state=json.dumps(request.cognitive_context.model_dump() if request.cognitive_context else {})
            )
            db.add(user_message)
            db.commit()

            # Get streaming response from OpenAI with optimizations
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
            client = get_openai_client()
            if not client:
                raise Exception("OpenAI API key not configured")

            stream = client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                stream=True,
                max_completion_tokens=500,
                # Performance optimizations
                temperature=0.7,  # Slightly reduce randomness for better caching potential
                top_p=0.9,        # Nucleus sampling for efficiency
            )

            full_response = ""
            token_buffer = ""
            buffer_size = 5  # Buffer tokens for smoother streaming

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    token_buffer += content

                    # Send buffered tokens for smoother streaming
                    if len(token_buffer) >= buffer_size or '\n' in token_buffer:
                        yield f"data: {json.dumps({'type': 'token', 'content': token_buffer})}\n\n"
                        token_buffer = ""

            # Send any remaining buffered content
            if token_buffer:
                yield f"data: {json.dumps({'type': 'token', 'content': token_buffer})}\n\n"

            # Save assistant response
            assistant_message = Message(
                conversation_id=conversation.id,
                role="assistant",
                content=full_response,
                cognitive_state=json.dumps({})
            )
            db.add(assistant_message)

            # Goal tracking and conversation analysis
            detected_goals = cognitive_processor.detect_goals_in_conversation(
                request.message, full_response, current_user.id, db
            )
            
            progress_updates = cognitive_processor.track_goal_progress(
                request.message, full_response, current_user.id, db
            )
            
            # Apply progress updates to goals
            for update in progress_updates:
                goal = db.query(Goal).filter(Goal.id == update["goal_id"]).first()
                if goal:
                    new_completion = max(0, min(100, goal.completion_percentage + update["completion_change"]))
                    if new_completion != goal.completion_percentage:
                        # Create progress update record
                        goal_update = GoalUpdate(
                            goal_id=goal.id,
                            user_id=current_user.id,
                            conversation_id=conversation.id,
                            update_type="progress",
                            content=f"Progress detected from conversation: {update['progress_type']} indicators",
                            previous_completion=goal.completion_percentage,
                            new_completion=new_completion
                        )
                        db.add(goal_update)
                        
                        goal.completion_percentage = new_completion
                        goal.last_discussed_at = datetime.now(timezone.utc)
                        goal.updated_at = datetime.now(timezone.utc)
            
            # Link conversation to detected goals
            if detected_goals:
                cognitive_processor.link_conversation_to_goals(
                    conversation.id, detected_goals, current_user.id, db
                )
            
            # Update cognitive state with advanced learning
            current_state = request.cognitive_context or CognitiveState()
            updated_state = cognitive_processor.update_cognitive_state(
                current_state, request.message, full_response, current_user.id, db
            )

            # Create memory if conversation is meaningful
            if len(full_response) > 50 or any(keyword in request.message.lower() for keyword in 
                                           ['remember', 'important', 'feel', 'think', 'believe', 'love', 'hate']):
                memory_content = f"User said: '{request.message}' - I responded with insights about: {', '.join(updated_state.focus_areas[:3])}"
                importance = min(1.0, len(full_response) / 200 + len(updated_state.focus_areas) * 0.2)

                memory = Memory(
                    user_id=current_user.id,
                    content=memory_content,
                    category="conversation",
                    importance_score=importance
                )
                db.add(memory)

            # Update conversation timestamp
            conversation.updated_at = datetime.now(timezone.utc)
            db.commit()

            # Send cognitive update
            yield f"data: {json.dumps({'type': 'cognitive_update', 'state': updated_state.model_dump()})}\n\n"

            # Send completion signal
            yield f"data: {json.dumps({'type': 'complete', 'message': 'Response completed'})}\n\n"

        except Exception as e:
            logger.error(f"Error in stream_chat: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        generate_response(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

# File upload and multi-modal analysis endpoints
UPLOAD_DIRECTORY = Path("./uploads")
UPLOAD_DIRECTORY.mkdir(exist_ok=True)

@app.post("/api/upload/")
async def upload_file(file: UploadFile = File(...), current_user: User = Depends(get_current_user)):
    """Upload a file for analysis"""
    file_location = UPLOAD_DIRECTORY / f"{uuid.uuid4()}-{file.filename}"
    try:
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
        logger.info(f"File {file.filename} uploaded successfully to {file_location}")

        # Here you would typically trigger an analysis task
        # For now, we just return the file path and type
        return {
            "filename": file.filename,
            "content_type": file.content_type,
            "file_path": str(file_location),
            "message": "File uploaded. Analysis to be performed."
        }
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")
    finally:
        await file.close()

@app.get("/api/analyze/{file_path:path}")
async def analyze_file(file_path: str, current_user: User = Depends(get_current_user)):
    """Analyze an uploaded file (placeholder for actual analysis)"""
    full_path = Path(file_path)
    if not full_path.is_file() or not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Placeholder for multi-modal analysis
    # In a real application, you'd use libraries like Pillow for images,
    # Tika for documents, etc., and potentially OpenAI's Vision API.
    analysis_result = {
        "file_info": {
            "filename": full_path.name,
            "size": full_path.stat().st_size,
            "type": "unknown" # Determine based on file extension or content type
        },
        "analysis": "Analysis not yet implemented for this file type."
    }

    if full_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.gif']:
        analysis_result["analysis"] = "Image analysis would be performed here."
        analysis_result["file_info"]["type"] = "image"
    elif full_path.suffix.lower() in ['.pdf', '.txt', '.docx', '.doc']:
        analysis_result["analysis"] = "Document analysis would be performed here."
        analysis_result["file_info"]["type"] = "document"
    elif full_path.suffix.lower() in ['.py', '.js', '.cpp', '.java']:
        analysis_result["analysis"] = "Code analysis would be performed here."
        analysis_result["file_info"]["type"] = "code"

    return analysis_result

@app.get("/api/download/{file_path:path}")
async def download_file(file_path: str):
    """Download an uploaded file"""
    full_path = Path(file_path)
    if not full_path.is_file() or not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(full_path, filename=full_path.name)


# Collaboration endpoints
@app.post("/api/shared-conversations/")
async def create_shared_conversation(
    conversation_data: SharedConversationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new shared conversation with specified participants"""
    # TODO: Implement actual shared conversation creation logic, linking users, etc.
    logger.info(f"Creating shared conversation '{conversation_data.name}' with participants: {conversation_data.participant_usernames}")
    return {"message": "Shared conversation creation endpoint reached. Implementation pending."}

@app.post("/api/shared-conversations/{shared_conversation_id}/messages/")
async def send_shared_message(
    shared_conversation_id: int,
    message_data: SharedMessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Send a message to a shared conversation"""
    # TODO: Implement message sending logic, including broadcasting to participants
    logger.info(f"Sending message to shared conversation {shared_conversation_id}: {message_data.content}")
    return {"message": "Shared message sending endpoint reached. Implementation pending."}

@app.post("/api/shared-conversations/{shared_conversation_id}/typing/")
async def update_typing_status(
    shared_conversation_id: int,
    status_data: TypingStatus,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update typing status for a shared conversation"""
    # TODO: Implement logic to broadcast typing status to other participants
    logger.info(f"User {current_user.username} typing status in shared conversation {shared_conversation_id}: {status_data.is_typing}")
    return {"message": "Typing status update endpoint reached. Implementation pending."}


@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time cognitive updates and collaboration"""
    await ws_manager.connect(websocket, client_id)

    try:
        while True:
            # Listen for messages from client
            data = await websocket.receive_json()

            if data.get("type") == "ping":
                # Respond to ping with pong
                await websocket.send_json({"type": "pong", "timestamp": datetime.now(timezone.utc).isoformat()})

            elif data.get("type") == "cognitive_sync":
                # Sync cognitive state
                if client_id in ws_manager.cognitive_states:
                    await websocket.send_json({
                        "type": "cognitive_state",
                        "state": ws_manager.cognitive_states[client_id].model_dump()
                    })
            # Handle collaboration messages via WebSocket
            elif data.get("type") == "shared_message":
                # Placeholder: Process and broadcast shared messages
                pass
            elif data.get("type") == "typing_update":
                # Placeholder: Process and broadcast typing updates
                pass

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        ws_manager.disconnect(websocket, client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="localhost",
        port=8000,
        reload=True,
        log_level="info"
    )