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

class ContradictionDetector:
    """Detects contradictions in user statements and helps with belief revision"""
    
    def __init__(self):
        # Common cognitive distortion patterns
        self.distortion_patterns = {
            "all_or_nothing": {
                "indicators": ["always", "never", "everything", "nothing", "everyone", "no one", "all", "none"],
                "context_negative": ["fail", "wrong", "bad", "hate", "terrible", "awful", "disaster"],
                "description": "Thinking in absolute terms without recognizing middle ground"
            },
            "overgeneralization": {
                "indicators": ["always happens", "never works", "every time", "typical", "story of my life"],
                "context_negative": ["fail", "goes wrong", "doesn't work", "mess up"],
                "description": "Drawing broad conclusions from single events"
            },
            "self_deprecation": {
                "indicators": ["i'm terrible", "i'm awful", "i'm not good", "i can't do", "i'm useless", "i'm hopeless"],
                "contradicts_with": ["accomplishment", "success", "achievement", "completed", "finished", "proud"],
                "description": "Consistently putting oneself down despite evidence to the contrary"
            },
            "discounting_positives": {
                "indicators": ["but", "however", "it was just", "anyone could", "it doesn't count", "it was luck"],
                "context_positive": ["success", "achievement", "compliment", "praise", "good job", "well done"],
                "description": "Minimizing positive experiences or achievements"
            },
            "emotional_reasoning": {
                "indicators": ["i feel like", "since i feel", "because i'm", "i must be"],
                "context_emotions": ["stupid", "worthless", "failure", "useless", "incompetent"],
                "description": "Believing that feelings reflect reality without evidence"
            }
        }
        
        # Contradiction detection patterns
        self.contradiction_indicators = {
            "ability_vs_accomplishment": {
                "negative_claims": ["i'm not good at", "i can't", "i'm terrible at", "i'm bad with"],
                "positive_evidence": ["completed", "finished", "achieved", "succeeded", "got promoted", "was chosen", "won", "passed"]
            },
            "social_vs_relationships": {
                "negative_claims": ["no one likes me", "i have no friends", "everyone hates me", "i'm unlikeable"],
                "positive_evidence": ["friend said", "invited to", "people tell me", "colleagues", "partner", "family"]
            },
            "value_vs_contributions": {
                "negative_claims": ["i'm worthless", "i don't matter", "i'm useless", "i contribute nothing"],
                "positive_evidence": ["helped", "supported", "made difference", "thanked", "appreciated", "needed"]
            },
            "intelligence_vs_problem_solving": {
                "negative_claims": ["i'm stupid", "i'm not smart", "i'm dumb", "i can't think"],
                "positive_evidence": ["figured out", "solved", "learned", "understood", "analyzed", "explained"]
            }
        }
        
        # Growth indicators to track user development
        self.growth_indicators = {
            "confidence": ["i think i can", "i'll try", "i believe", "i'm willing", "i'm getting better"],
            "self_awareness": ["i realize", "i've learned", "i now understand", "i see that", "i've grown"],
            "positive_self_talk": ["i'm proud", "i did well", "i'm improving", "i'm capable", "i can handle"],
            "resilience": ["i bounced back", "i overcame", "i persevered", "i didn't give up", "i kept going"],
            "boundaries": ["i said no", "i stood up", "i protected myself", "i set limits", "i chose myself"]
        }

    def detect_contradictions(self, current_message: str, user_id: int, db: Session) -> List[Dict]:
        """Detect contradictions between current statements and past conversations"""
        from database import Message, Conversation
        
        current_lower = current_message.lower()
        contradictions = []
        
        # Get recent user messages (last 30 days)
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.updated_at > thirty_days_ago
        ).all()
        
        past_messages = []
        for conv in conversations:
            messages = db.query(Message).filter(
                Message.conversation_id == conv.id,
                Message.role == "user"
            ).all()
            past_messages.extend([msg.content.lower() for msg in messages])
        
        # Check for each contradiction pattern
        for pattern_name, pattern in self.contradiction_indicators.items():
            # Check if current message contains negative claims
            current_has_negative = any(claim in current_lower for claim in pattern["negative_claims"])
            
            if current_has_negative:
                # Look for positive evidence in past messages
                positive_evidence = []
                for msg in past_messages:
                    for evidence in pattern["positive_evidence"]:
                        if evidence in msg:
                            positive_evidence.append(msg[:100])  # Store snippet
                            break
                
                if positive_evidence:
                    contradictions.append({
                        "type": pattern_name,
                        "current_claim": [claim for claim in pattern["negative_claims"] if claim in current_lower][0],
                        "contradicting_evidence": positive_evidence[:3],  # Top 3 examples
                        "pattern_description": self._get_contradiction_description(pattern_name),
                        "confidence": min(len(positive_evidence) / 3, 1.0),
                        "gentle_response": self._generate_gentle_contradiction_response(pattern_name, positive_evidence)
                    })
        
        return contradictions

    def detect_cognitive_distortions(self, message: str) -> List[Dict]:
        """Detect cognitive distortions in user's message"""
        message_lower = message.lower()
        distortions = []
        
        for distortion_name, pattern in self.distortion_patterns.items():
            distortion_score = 0
            found_indicators = []
            
            # Check for primary indicators
            for indicator in pattern["indicators"]:
                if indicator in message_lower:
                    distortion_score += 1
                    found_indicators.append(indicator)
            
            # Check for context-specific patterns
            if distortion_score > 0:
                if "context_negative" in pattern:
                    negative_context = any(neg in message_lower for neg in pattern["context_negative"])
                    if negative_context:
                        distortion_score += 0.5
                
                if "context_positive" in pattern:
                    positive_context = any(pos in message_lower for pos in pattern["context_positive"])
                    if positive_context:
                        distortion_score += 0.5
                
                if "context_emotions" in pattern:
                    emotional_context = any(emo in message_lower for emo in pattern["context_emotions"])
                    if emotional_context:
                        distortion_score += 0.5
                
                if distortion_score >= 1.0:  # Threshold for detection
                    distortions.append({
                        "type": distortion_name,
                        "description": pattern["description"],
                        "indicators_found": found_indicators,
                        "confidence": min(distortion_score / 2, 1.0),
                        "gentle_response": self._generate_distortion_response(distortion_name)
                    })
        
        return distortions

    def detect_growth_patterns(self, message: str, user_id: int, db: Session) -> List[Dict]:
        """Detect signs of personal growth and positive changes"""
        message_lower = message.lower()
        growth_detected = []
        
        for growth_type, indicators in self.growth_indicators.items():
            found_indicators = [indicator for indicator in indicators if indicator in message_lower]
            
            if found_indicators:
                # Check if this represents genuine growth by comparing to past patterns
                past_growth = self._check_past_growth_mentions(user_id, growth_type, db)
                
                growth_detected.append({
                    "type": growth_type,
                    "indicators": found_indicators,
                    "is_new_growth": len(past_growth) < 2,  # Less than 2 mentions before
                    "past_mentions": len(past_growth),
                    "encouragement_response": self._generate_growth_encouragement(growth_type, found_indicators)
                })
        
        return growth_detected

    def _check_past_growth_mentions(self, user_id: int, growth_type: str, db: Session) -> List[str]:
        """Check how often this type of growth has been mentioned before"""
        from database import Message, Conversation
        
        # Get messages from last 60 days
        sixty_days_ago = datetime.now(timezone.utc) - timedelta(days=60)
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.updated_at > sixty_days_ago
        ).all()
        
        growth_mentions = []
        indicators = self.growth_indicators[growth_type]
        
        for conv in conversations:
            messages = db.query(Message).filter(
                Message.conversation_id == conv.id,
                Message.role == "user"
            ).all()
            
            for msg in messages:
                if any(indicator in msg.content.lower() for indicator in indicators):
                    growth_mentions.append(msg.content[:100])
        
        return growth_mentions

    def update_user_beliefs(self, user_id: int, message: str, contradictions: List[Dict], growth_patterns: List[Dict], db: Session):
        """Update stored beliefs about the user based on contradictions and growth"""
        from database import UserPreference
        
        # Update beliefs based on contradictions
        for contradiction in contradictions:
            if contradiction["confidence"] > 0.7:
                # Store updated belief
                belief_type = f"self_perception_{contradiction['type']}"
                
                # Create or update preference indicating more balanced self-view
                preference = db.query(UserPreference).filter(
                    UserPreference.user_id == user_id,
                    UserPreference.preference_type == belief_type,
                    UserPreference.preference_value == "balanced_view"
                ).first()
                
                if preference:
                    preference.confidence_score = min(1.0, preference.confidence_score + 0.1)
                    preference.updated_at = datetime.now(timezone.utc)
                else:
                    preference = UserPreference(
                        user_id=user_id,
                        preference_type=belief_type,
                        preference_value="balanced_view",
                        confidence_score=0.3,
                        learned_from="contradiction_detection"
                    )
                    db.add(preference)
        
        # Update beliefs based on growth patterns
        for growth in growth_patterns:
            if growth["is_new_growth"]:
                belief_type = f"growth_area_{growth['type']}"
                
                preference = UserPreference(
                    user_id=user_id,
                    preference_type=belief_type,
                    preference_value="showing_progress",
                    confidence_score=0.5,
                    learned_from="growth_detection"
                )
                db.add(preference)
        
        db.commit()

    def _get_contradiction_description(self, pattern_name: str) -> str:
        """Get human-readable description of contradiction pattern"""
        descriptions = {
            "ability_vs_accomplishment": "Claims of inability despite past accomplishments",
            "social_vs_relationships": "Claims of social rejection despite relationship evidence",
            "value_vs_contributions": "Claims of worthlessness despite meaningful contributions",
            "intelligence_vs_problem_solving": "Claims of stupidity despite problem-solving evidence"
        }
        return descriptions.get(pattern_name, "Inconsistency in self-perception")

    def _generate_gentle_contradiction_response(self, pattern_name: str, evidence: List[str]) -> str:
        """Generate a gentle response pointing out the contradiction"""
        responses = {
            "ability_vs_accomplishment": [
                "I've noticed you often say you're not good at things, but you've shared several accomplishments with me. ",
                "You mention struggling with your abilities, yet I remember you telling me about times you succeeded. ",
                "I hear you saying you can't do something, but haven't you achieved similar things before? "
            ],
            "social_vs_relationships": [
                "You say no one likes you, but you've mentioned friends and positive social interactions. ",
                "I notice you feel unlikeable, yet you've shared stories about people caring for you. ",
                "You feel socially rejected, but you've told me about meaningful connections. "
            ],
            "value_vs_contributions": [
                "You say you're worthless, but you've shared examples of how you've helped others. ",
                "I hear you feeling like you don't matter, yet you've told me about your positive impact. ",
                "You feel useless, but haven't you described ways you've contributed and made a difference? "
            ],
            "intelligence_vs_problem_solving": [
                "You call yourself stupid, but you've shown me clever problem-solving many times. ",
                "I notice you doubt your intelligence, yet you've shared insights that show real understanding. ",
                "You feel dumb, but you've described figuring out complex situations. "
            ]
        }
        
        base_responses = responses.get(pattern_name, ["I notice some inconsistency in how you see yourself. "])
        base = base_responses[0]  # Use first response
        
        return f"{base}Could we explore this together? Sometimes our inner critic is louder than our inner truth."

    def _generate_distortion_response(self, distortion_name: str) -> str:
        """Generate helpful response for cognitive distortions"""
        responses = {
            "all_or_nothing": "I notice some 'all or nothing' thinking. Life usually has more gray areas than we realize. What might be somewhere in between?",
            "overgeneralization": "It sounds like you're taking one situation and applying it broadly. Could this be more specific to this particular case?",
            "self_deprecation": "I hear you being quite hard on yourself. If a friend told you this same thing, what would you say to them?",
            "discounting_positives": "I notice you might be minimizing something positive. What if we gave this achievement the credit it deserves?",
            "emotional_reasoning": "Your feelings are valid, but they might not reflect the complete picture. What evidence supports or challenges this feeling?"
        }
        return responses.get(distortion_name, "I wonder if there might be another way to look at this situation?")

    def _generate_growth_encouragement(self, growth_type: str, indicators: List[str]) -> str:
        """Generate encouraging response for detected growth"""
        encouragements = {
            "confidence": "I'm hearing more confidence in your voice! This is wonderful growth.",
            "self_awareness": "Your self-awareness is really developing. That's such valuable insight.",
            "positive_self_talk": "I love hearing you speak kindly about yourself. That's a beautiful shift.",
            "resilience": "Your resilience is really showing. You're handling challenges with more strength.",
            "boundaries": "Setting boundaries shows real self-respect. That's powerful personal growth."
        }
        
        base = encouragements.get(growth_type, "I'm noticing positive changes in how you approach things.")
        return f"{base} How does this feel different for you?"

class EmotionalTrajectoryTracker:
    """Advanced emotional trajectory tracking and pattern recognition for mental health support"""
    
    def __init__(self):
        # Enhanced emotional analysis
        self.emotion_categories = {
            "anxiety": ["anxious", "worried", "nervous", "stressed", "overwhelmed", "panic", "tense", "uneasy"],
            "depression": ["sad", "depressed", "hopeless", "empty", "worthless", "lonely", "numb", "despair"],
            "anger": ["angry", "frustrated", "irritated", "furious", "rage", "annoyed", "bitter", "resentful"],
            "joy": ["happy", "joyful", "excited", "elated", "cheerful", "content", "delighted", "euphoric"],
            "fear": ["scared", "afraid", "terrified", "fearful", "paranoid", "phobic", "dread", "alarmed"],
            "disgust": ["disgusted", "revolted", "repulsed", "sickened", "appalled", "nauseated"],
            "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "bewildered"],
            "shame": ["ashamed", "guilty", "embarrassed", "humiliated", "mortified", "regretful"],
            "love": ["love", "affection", "caring", "compassionate", "tender", "devoted", "adoring"],
            "hope": ["hopeful", "optimistic", "confident", "encouraged", "positive", "motivated"],
            "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil", "composed", "centered"],
            "confusion": ["confused", "puzzled", "uncertain", "lost", "unclear", "bewildered", "perplexed"]
        }
        
        # Mental health warning indicators
        self.warning_patterns = {
            "escalating_distress": {
                "emotions": ["anxious", "overwhelmed", "stressed", "panicked"],
                "frequency_threshold": 3,  # mentions in 7 days
                "severity_threshold": 0.7
            },
            "persistent_sadness": {
                "emotions": ["sad", "depressed", "hopeless", "empty", "worthless"],
                "frequency_threshold": 5,  # mentions in 14 days
                "severity_threshold": 0.6
            },
            "isolation_indicators": {
                "emotions": ["lonely", "isolated", "disconnected", "alone"],
                "frequency_threshold": 3,  # mentions in 7 days
                "severity_threshold": 0.5
            },
            "anger_escalation": {
                "emotions": ["angry", "furious", "rage", "hatred"],
                "frequency_threshold": 4,  # mentions in 7 days
                "severity_threshold": 0.8
            },
            "self_harm_ideation": {
                "keywords": ["hurt myself", "end it all", "not worth living", "give up", "can't go on"],
                "frequency_threshold": 1,  # immediate attention
                "severity_threshold": 0.9
            }
        }
        
        # Helpful intervention topics by emotional state
        self.helpful_interventions = {
            "anxiety": ["breathing exercises", "mindfulness", "grounding techniques", "relaxation", "stress management"],
            "depression": ["self-care", "positive activities", "social connection", "professional help", "routine building"],
            "anger": ["anger management", "cooling down", "communication skills", "conflict resolution", "stress relief"],
            "overwhelmed": ["time management", "prioritization", "breaking tasks down", "delegation", "boundaries"],
            "lonely": ["social activities", "community", "reaching out", "support groups", "friendship"]
        }
        
        # Topics to avoid or handle carefully
        self.sensitive_topics = {
            "avoid_when_depressed": ["major life decisions", "productivity pressure", "comparison with others"],
            "avoid_when_anxious": ["worst-case scenarios", "uncertainty", "overwhelming information"],
            "handle_carefully": ["relationship breakups", "job loss", "family conflict", "health issues", "financial stress"]
        }

    def analyze_emotional_trajectory(self, user_id: int, db: Session, days_back: int = 30) -> Dict:
        """Analyze user's emotional trajectory over time"""
        from database import EmotionalState
        
        # Get emotional states from the specified period
        since_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        emotional_states = db.query(EmotionalState).filter(
            EmotionalState.user_id == user_id,
            EmotionalState.detected_at >= since_date
        ).order_by(EmotionalState.detected_at.asc()).all()
        
        if not emotional_states:
            return {"status": "insufficient_data", "message": "Not enough emotional data to analyze trajectory"}
        
        # Calculate trajectory metrics
        trajectory_data = []
        emotion_frequencies = {}
        intensity_over_time = []
        valence_over_time = []
        
        for state in emotional_states:
            trajectory_data.append({
                "date": state.detected_at,
                "emotion": state.primary_emotion,
                "intensity": state.emotional_intensity,
                "valence": state.valence,
                "arousal": state.arousal
            })
            
            # Count emotion frequencies
            emotion_frequencies[state.primary_emotion] = emotion_frequencies.get(state.primary_emotion, 0) + 1
            intensity_over_time.append(state.emotional_intensity)
            valence_over_time.append(state.valence)
        
        # Calculate trends
        avg_intensity = sum(intensity_over_time) / len(intensity_over_time)
        avg_valence = sum(valence_over_time) / len(valence_over_time)
        
        # Detect intensity trend (increasing/decreasing)
        if len(intensity_over_time) >= 5:
            recent_intensity = sum(intensity_over_time[-5:]) / 5
            older_intensity = sum(intensity_over_time[:5]) / 5
            intensity_trend = "increasing" if recent_intensity > older_intensity + 0.1 else "decreasing" if recent_intensity < older_intensity - 0.1 else "stable"
        else:
            intensity_trend = "stable"
        
        # Detect valence trend
        if len(valence_over_time) >= 5:
            recent_valence = sum(valence_over_time[-5:]) / 5
            older_valence = sum(valence_over_time[:5]) / 5
            valence_trend = "improving" if recent_valence > older_valence + 0.1 else "declining" if recent_valence < older_valence - 0.1 else "stable"
        else:
            valence_trend = "stable"
        
        return {
            "trajectory_data": trajectory_data,
            "emotion_frequencies": emotion_frequencies,
            "avg_intensity": avg_intensity,
            "avg_valence": avg_valence,
            "intensity_trend": intensity_trend,
            "valence_trend": valence_trend,
            "total_entries": len(emotional_states),
            "dominant_emotions": sorted(emotion_frequencies.items(), key=lambda x: x[1], reverse=True)[:5]
        }

    def detect_emotional_patterns(self, user_id: int, db: Session) -> List[Dict]:
        """Detect concerning emotional patterns"""
        from database import EmotionalState, EmotionalPattern
        
        detected_patterns = []
        
        # Get recent emotional states (last 30 days)
        since_date = datetime.now(timezone.utc) - timedelta(days=30)
        emotional_states = db.query(EmotionalState).filter(
            EmotionalState.user_id == user_id,
            EmotionalState.detected_at >= since_date
        ).order_by(EmotionalState.detected_at.desc()).all()
        
        if len(emotional_states) < 5:
            return detected_patterns
        
        # Check for each warning pattern
        for pattern_name, pattern_config in self.warning_patterns.items():
            if pattern_name == "self_harm_ideation":
                # Special handling for self-harm indicators
                pattern_count = 0
                for state in emotional_states:
                    if state.emotional_context:
                        for keyword in pattern_config["keywords"]:
                            if keyword.lower() in state.emotional_context.lower():
                                pattern_count += 1
                                break
                
                if pattern_count >= pattern_config["frequency_threshold"]:
                    detected_patterns.append({
                        "pattern_type": "critical",
                        "pattern_name": pattern_name,
                        "severity": 1.0,  # Maximum severity
                        "frequency": pattern_count,
                        "description": "Critical: Self-harm ideation detected",
                        "requires_immediate_attention": True,
                        "recommended_actions": ["Encourage professional help", "Provide crisis resources", "Monitor closely"]
                    })
            else:
                # Check for emotional pattern frequencies
                pattern_emotions = pattern_config["emotions"]
                pattern_count = 0
                high_intensity_count = 0
                
                for state in emotional_states:
                    if state.primary_emotion in pattern_emotions:
                        pattern_count += 1
                        if state.emotional_intensity >= pattern_config["severity_threshold"]:
                            high_intensity_count += 1
                
                if pattern_count >= pattern_config["frequency_threshold"]:
                    severity = min(1.0, (pattern_count / pattern_config["frequency_threshold"]) * 0.5 + 
                                 (high_intensity_count / max(1, pattern_count)) * 0.5)
                    
                    detected_patterns.append({
                        "pattern_type": "warning",
                        "pattern_name": pattern_name,
                        "severity": severity,
                        "frequency": pattern_count,
                        "high_intensity_frequency": high_intensity_count,
                        "description": self._get_pattern_description(pattern_name, pattern_count),
                        "requires_immediate_attention": severity >= 0.8,
                        "recommended_actions": self._get_recommended_actions(pattern_name)
                    })
        
        # Sort by severity
        detected_patterns.sort(key=lambda x: x["severity"], reverse=True)
        return detected_patterns

    def _get_pattern_description(self, pattern_name: str, frequency: int) -> str:
        """Generate human-readable pattern descriptions"""
        descriptions = {
            "escalating_distress": f"Increasing anxiety and stress levels detected ({frequency} mentions recently)",
            "persistent_sadness": f"Ongoing signs of depression or persistent sadness ({frequency} mentions recently)",
            "isolation_indicators": f"Patterns of social isolation and loneliness ({frequency} mentions recently)",
            "anger_escalation": f"Escalating anger and frustration levels ({frequency} mentions recently)"
        }
        return descriptions.get(pattern_name, f"Concerning pattern detected: {pattern_name}")

    def _get_recommended_actions(self, pattern_name: str) -> List[str]:
        """Get recommended actions for each pattern type"""
        actions = {
            "escalating_distress": [
                "Explore stress management techniques",
                "Discuss what's causing the increased anxiety",
                "Consider grounding exercises or mindfulness",
                "Evaluate if professional support might be helpful"
            ],
            "persistent_sadness": [
                "Gently explore the source of sadness",
                "Encourage self-care activities",
                "Discuss social support systems",
                "Consider suggesting professional mental health support"
            ],
            "isolation_indicators": [
                "Encourage social connection",
                "Explore barriers to social interaction",
                "Suggest small steps to reconnect",
                "Discuss community or support group options"
            ],
            "anger_escalation": [
                "Explore healthy ways to express anger",
                "Discuss stress reduction techniques",
                "Help identify triggers",
                "Consider anger management strategies"
            ]
        }
        return actions.get(pattern_name, ["Monitor closely", "Provide supportive responses"])

    def create_emotional_state_entry(self, user_id: int, message: str, conversation_id: int, 
                                   message_id: int, db: Session) -> Optional[int]:
        """Create an emotional state entry from a message"""
        from database import EmotionalState
        
        # Analyze the emotional content
        emotional_analysis = self._analyze_message_emotions(message)
        
        if not emotional_analysis["emotions_detected"]:
            return None  # No significant emotions detected
        
        # Create emotional state entry
        emotional_state = EmotionalState(
            user_id=user_id,
            conversation_id=conversation_id,
            message_id=message_id,
            primary_emotion=emotional_analysis["primary_emotion"],
            emotional_intensity=emotional_analysis["intensity"],
            valence=emotional_analysis["valence"],
            arousal=emotional_analysis["arousal"],
            detected_emotions=emotional_analysis["all_emotions"],
            emotional_context=emotional_analysis["context"]
        )
        
        db.add(emotional_state)
        db.commit()
        db.refresh(emotional_state)
        
        return emotional_state.id

    def _analyze_message_emotions(self, message: str) -> Dict:
        """Analyze emotions in a message"""
        message_lower = message.lower()
        detected_emotions = []
        emotion_scores = {}
        
        # Check for emotions in each category
        for category, emotions in self.emotion_categories.items():
            category_score = 0
            found_emotions = []
            
            for emotion in emotions:
                if emotion in message_lower:
                    found_emotions.append(emotion)
                    category_score += 1
            
            if found_emotions:
                detected_emotions.extend(found_emotions)
                emotion_scores[category] = category_score / len(emotions)
        
        if not detected_emotions:
            return {"emotions_detected": False}
        
        # Determine primary emotion (highest scoring category)
        primary_category = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        primary_emotion = primary_category
        
        # Calculate metrics
        intensity = min(1.0, len(detected_emotions) * 0.3)  # Intensity based on number of emotions
        
        # Calculate valence (positive/negative)
        positive_categories = ["joy", "love", "hope", "calm"]
        negative_categories = ["anxiety", "depression", "anger", "fear", "disgust", "shame"]
        
        positive_score = sum(emotion_scores.get(cat, 0) for cat in positive_categories)
        negative_score = sum(emotion_scores.get(cat, 0) for cat in negative_categories)
        
        if positive_score + negative_score > 0:
            valence = (positive_score - negative_score) / (positive_score + negative_score)
        else:
            valence = 0.0
        
        # Calculate arousal (excited/calm)
        high_arousal_categories = ["anger", "fear", "joy", "surprise"]
        low_arousal_categories = ["depression", "calm", "shame"]
        
        high_arousal_score = sum(emotion_scores.get(cat, 0) for cat in high_arousal_categories)
        low_arousal_score = sum(emotion_scores.get(cat, 0) for cat in low_arousal_categories)
        
        if high_arousal_score + low_arousal_score > 0:
            arousal = high_arousal_score / (high_arousal_score + low_arousal_score)
        else:
            arousal = 0.5
        
        return {
            "emotions_detected": True,
            "primary_emotion": primary_emotion,
            "all_emotions": detected_emotions,
            "intensity": intensity,
            "valence": valence,
            "arousal": arousal,
            "context": message[:200],  # First 200 chars for context
            "emotion_scores": emotion_scores
        }

    def get_helpful_topics(self, user_id: int, db: Session) -> List[str]:
        """Get topics that tend to help this specific user"""
        from database import EmotionalState, Message
        
        # Find conversations where user's emotional state improved
        helpful_topics = []
        
        # Get emotional states that improved (went from negative to positive valence)
        emotional_states = db.query(EmotionalState).filter(
            EmotionalState.user_id == user_id
        ).order_by(EmotionalState.detected_at.desc()).limit(100).all()
        
        for i in range(1, len(emotional_states)):
            current_state = emotional_states[i-1]  # More recent
            previous_state = emotional_states[i]   # Earlier
            
            # Check if there was improvement
            if (current_state.valence > previous_state.valence + 0.2 or 
                current_state.emotional_intensity < previous_state.emotional_intensity - 0.2):
                
                # Find the conversation topics around this improvement
                if current_state.conversation_id:
                    messages = db.query(Message).filter(
                        Message.conversation_id == current_state.conversation_id,
                        Message.role == "assistant",
                        Message.timestamp <= current_state.detected_at
                    ).order_by(Message.timestamp.desc()).limit(3).all()
                    
                    for message in messages:
                        # Extract key topics from helpful assistant messages
                        content_lower = message.content.lower()
                        for category, topics in self.helpful_interventions.items():
                            for topic in topics:
                                if topic in content_lower and topic not in helpful_topics:
                                    helpful_topics.append(topic)
        
        return helpful_topics[:10]  # Return top 10 helpful topics

    def get_topics_to_avoid(self, current_emotional_state: str) -> List[str]:
        """Get topics to avoid based on current emotional state"""
        topics_to_avoid = []
        
        if current_emotional_state in ["sad", "depressed", "hopeless"]:
            topics_to_avoid.extend(self.sensitive_topics["avoid_when_depressed"])
        
        if current_emotional_state in ["anxious", "worried", "stressed"]:
            topics_to_avoid.extend(self.sensitive_topics["avoid_when_anxious"])
        
        topics_to_avoid.extend(self.sensitive_topics["handle_carefully"])
        
        return list(set(topics_to_avoid))  # Remove duplicates

class HypothesisFormationEngine:
    """Advanced hypothesis formation and meta-memory system for understanding user patterns and preferences"""
    
    def __init__(self):
        # Hypothesis formation patterns
        self.behavioral_patterns = {
            "ux_preference": {
                "indicators": ["user experience", "ease of use", "intuitive", "simple", "clean", "elegant"],
                "context_positive": ["love", "prefer", "like", "choose", "favor"],
                "context_negative": ["hate", "dislike", "avoid", "frustrating", "confusing"],
                "hypothesis_templates": [
                    "User prioritizes {preference_area} over {alternative_area}",
                    "User is drawn to solutions that emphasize {key_values}",
                    "User shows consistent preference for {pattern_type} approaches"
                ]
            },
            "technical_complexity": {
                "indicators": ["simple", "complex", "technical", "advanced", "basic", "sophisticated"],
                "context_preference": ["prefer", "like", "choose", "lean toward", "gravitate"],
                "context_avoidance": ["avoid", "skip", "ignore", "overwhelmed by", "intimidated"],
                "hypothesis_templates": [
                    "User prefers {complexity_level} technical solutions",
                    "User values {approach_type} over technical sophistication",
                    "User's comfort zone appears to be {technical_level}"
                ]
            },
            "problem_solving_style": {
                "indicators": ["systematic", "creative", "analytical", "intuitive", "methodical", "experimental"],
                "context_markers": ["approach", "tackle", "solve", "handle", "deal with"],
                "hypothesis_templates": [
                    "User tends to approach problems in a {style_type} manner",
                    "User's problem-solving style emphasizes {key_characteristics}",
                    "User shows preference for {methodology} approaches to challenges"
                ]
            },
            "collaboration_style": {
                "indicators": ["team", "solo", "independent", "collaborative", "group", "alone"],
                "context_work": ["work", "project", "task", "assignment", "collaboration"],
                "hypothesis_templates": [
                    "User thrives in {environment_type} working environments",
                    "User's collaboration style tends toward {approach_type}",
                    "User shows preference for {interaction_level} project approaches"
                ]
            }
        }
        
        # Evolution catalyst patterns
        self.evolution_catalysts = {
            "frustration": {
                "triggers": ["frustrated", "annoying", "doesn't work", "failing", "broken", "difficult"],
                "context": ["with", "about", "when", "during", "while"],
                "evolution_indicators": ["so I thought", "maybe we could", "what if", "I wonder if"]
            },
            "inspiration": {
                "triggers": ["saw", "noticed", "realized", "discovered", "found", "learned"],
                "context": ["that", "how", "when", "where", "while"],
                "evolution_indicators": ["made me think", "gave me an idea", "inspired me", "sparked"]
            },
            "constraint": {
                "triggers": ["can't", "won't work", "limited", "restricted", "blocked", "prevented"],
                "context": ["because", "since", "due to", "given that"],
                "evolution_indicators": ["so instead", "alternative", "different approach", "work around"]
            },
            "opportunity": {
                "triggers": ["could", "might", "possibility", "chance", "opportunity", "potential"],
                "context": ["to", "for", "with", "in", "through"],
                "evolution_indicators": ["build on", "expand", "develop", "explore further"]
            }
        }
        
        # Meta-memory tracking patterns
        self.memory_evolution_markers = {
            "refinement": ["actually", "more precisely", "to clarify", "what I really meant"],
            "expansion": ["also", "additionally", "building on that", "expanding on"],
            "correction": ["actually", "I was wrong", "let me correct", "that's not quite right"],
            "synthesis": ["combining", "bringing together", "connecting", "merging"],
            "breakthrough": ["ah!", "I see", "that's it", "breakthrough", "eureka", "clicked"]
        }

    def analyze_user_patterns(self, user_id: int, db: Session, conversation_window: int = 10) -> List[Dict]:
        """Analyze user patterns to form hypotheses about preferences and behaviors"""
        from database import Message, Conversation
        
        # Get recent conversations
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc()).limit(conversation_window).all()
        
        all_user_messages = []
        for conv in conversations:
            messages = db.query(Message).filter(
                Message.conversation_id == conv.id,
                Message.role == "user"
            ).all()
            all_user_messages.extend([(msg.content, conv.id, msg.id) for msg in messages])
        
        detected_patterns = []
        
        # Analyze each behavioral pattern type
        for pattern_name, pattern_config in self.behavioral_patterns.items():
            pattern_evidence = []
            preference_indicators = []
            
            for message_content, conv_id, msg_id in all_user_messages:
                content_lower = message_content.lower()
                
                # Check for pattern indicators
                found_indicators = [ind for ind in pattern_config["indicators"] if ind in content_lower]
                
                if found_indicators:
                    # Check for preference context
                    positive_context = any(pos in content_lower for pos in pattern_config.get("context_positive", []))
                    negative_context = any(neg in content_lower for neg in pattern_config.get("context_negative", []))
                    
                    if positive_context or negative_context:
                        evidence_item = {
                            "conversation_id": conv_id,
                            "message_id": msg_id,
                            "indicators": found_indicators,
                            "context": "positive" if positive_context else "negative",
                            "quote": message_content[:100] + "..." if len(message_content) > 100 else message_content,
                            "full_content": message_content
                        }
                        pattern_evidence.append(evidence_item)
                        
                        # Extract specific preference
                        for indicator in found_indicators:
                            preference_indicators.append({
                                "indicator": indicator,
                                "context": evidence_item["context"],
                                "strength": 1.0 if positive_context else -1.0
                            })
            
            # Generate hypothesis if sufficient evidence
            if len(pattern_evidence) >= 2:  # Minimum evidence threshold
                confidence = min(1.0, len(pattern_evidence) / 5)  # Confidence based on evidence count
                
                # Determine dominant preference
                positive_count = len([p for p in preference_indicators if p["strength"] > 0])
                negative_count = len([p for p in preference_indicators if p["strength"] < 0])
                
                if positive_count > negative_count:
                    preference_direction = "positive"
                    dominant_indicators = [p["indicator"] for p in preference_indicators if p["strength"] > 0]
                else:
                    preference_direction = "negative"
                    dominant_indicators = [p["indicator"] for p in preference_indicators if p["strength"] < 0]
                
                # Generate hypothesis text
                hypothesis_template = pattern_config["hypothesis_templates"][0]
                hypothesis_text = self._format_hypothesis(hypothesis_template, pattern_name, dominant_indicators, preference_direction)
                
                detected_patterns.append({
                    "pattern_type": pattern_name,
                    "hypothesis": hypothesis_text,
                    "confidence": confidence,
                    "evidence_count": len(pattern_evidence),
                    "evidence": pattern_evidence[:3],  # Top 3 pieces of evidence
                    "dominant_indicators": dominant_indicators[:3],
                    "preference_direction": preference_direction
                })
        
        return detected_patterns

    def track_idea_evolution(self, current_message: str, conversation_id: int, user_id: int, db: Session) -> Optional[Dict]:
        """Track how ideas evolve through conversations"""
        from database import Message, IdeaEvolution
        
        message_lower = current_message.lower()
        
        # Check for evolution markers
        detected_evolution = None
        catalyst_type = None
        catalyst_description = None
        
        # Detect catalyst type
        for catalyst, catalyst_config in self.evolution_catalysts.items():
            trigger_found = any(trigger in message_lower for trigger in catalyst_config["triggers"])
            evolution_found = any(evo in message_lower for evo in catalyst_config["evolution_indicators"])
            
            if trigger_found and evolution_found:
                catalyst_type = catalyst
                catalyst_description = self._extract_catalyst_description(current_message, catalyst_config)
                break
        
        # Check for meta-memory evolution markers
        evolution_stage = "development"  # Default
        for stage, markers in self.memory_evolution_markers.items():
            if any(marker in message_lower for marker in markers):
                evolution_stage = stage
                break
        
        if catalyst_type:
            # Extract the evolving idea
            idea_content = self._extract_idea_content(current_message, catalyst_type)
            
            if idea_content:
                # Check if this builds on previous ideas
                previous_ideas = db.query(IdeaEvolution).filter(
                    IdeaEvolution.user_id == user_id
                ).order_by(IdeaEvolution.evolved_at.desc()).limit(5).all()
                
                influenced_by = []
                for prev_idea in previous_ideas:
                    # Simple keyword matching to detect influence
                    prev_keywords = set(prev_idea.current_form.lower().split())
                    current_keywords = set(idea_content.lower().split())
                    overlap = len(prev_keywords.intersection(current_keywords))
                    
                    if overlap >= 3:  # Minimum keyword overlap
                        influenced_by.append({
                            "idea_id": prev_idea.id,
                            "title": prev_idea.idea_title,
                            "overlap_score": overlap / max(len(prev_keywords), len(current_keywords))
                        })
                
                detected_evolution = {
                    "idea_title": self._generate_idea_title(idea_content),
                    "current_form": idea_content,
                    "evolution_stage": evolution_stage,
                    "catalyst_type": catalyst_type,
                    "catalyst_description": catalyst_description,
                    "catalyst_quote": self._extract_catalyst_quote(current_message, catalyst_type),
                    "influenced_by": influenced_by[:3],  # Top 3 influences
                    "novelty_score": self._calculate_novelty_score(idea_content, previous_ideas),
                    "complexity_score": self._calculate_complexity_score(idea_content),
                    "feasibility_score": self._calculate_feasibility_score(idea_content)
                }
        
        return detected_evolution

    def generate_testable_hypotheses(self, user_patterns: List[Dict], user_id: int, db: Session) -> List[Dict]:
        """Generate testable hypotheses about user preferences and behaviors"""
        testable_hypotheses = []
        
        for pattern in user_patterns:
            if pattern["confidence"] > 0.6:  # Only generate tests for confident patterns
                # Create test suggestions based on pattern type
                test_suggestions = self._generate_test_suggestions(pattern)
                
                # Determine if this hypothesis already exists
                from database import Hypothesis
                existing = db.query(Hypothesis).filter(
                    Hypothesis.user_id == user_id,
                    Hypothesis.hypothesis_type == pattern["pattern_type"],
                    Hypothesis.status == "active"
                ).first()
                
                if not existing:
                    testable_hypotheses.append({
                        "title": f"User Preference: {pattern['pattern_type'].replace('_', ' ').title()}",
                        "description": pattern["hypothesis"],
                        "hypothesis_type": "preference",
                        "domain": pattern["pattern_type"],
                        "confidence_score": pattern["confidence"],
                        "initial_evidence": pattern["evidence"],
                        "test_suggestions": test_suggestions,
                        "is_testable": len(test_suggestions) > 0
                    })
        
        return testable_hypotheses

    def _format_hypothesis(self, template: str, pattern_type: str, indicators: List[str], direction: str) -> str:
        """Format hypothesis text using template and extracted data"""
        # Simple template formatting
        if "ux_preference" in pattern_type:
            if direction == "positive":
                return f"User prioritizes user experience and simplicity over technical complexity"
            else:
                return f"User shows tolerance for complexity when it serves functionality"
        
        elif "technical_complexity" in pattern_type:
            if direction == "positive":
                return f"User prefers straightforward, accessible technical solutions"
            else:
                return f"User is comfortable with advanced technical approaches"
        
        elif "problem_solving_style" in pattern_type:
            style_desc = ", ".join(indicators[:2]) if indicators else "systematic"
            return f"User tends to approach problems in a {style_desc} manner"
        
        elif "collaboration_style" in pattern_type:
            collab_style = indicators[0] if indicators else "collaborative"
            return f"User shows preference for {collab_style} working approaches"
        
        return template.format(
            preference_area=indicators[0] if indicators else "unknown",
            key_values=", ".join(indicators[:2]),
            pattern_type=pattern_type.replace("_", " ")
        )

    def _extract_catalyst_description(self, message: str, catalyst_config: Dict) -> str:
        """Extract what specifically triggered the idea evolution"""
        # Find sentences containing trigger words
        sentences = message.split('.')
        for sentence in sentences:
            if any(trigger in sentence.lower() for trigger in catalyst_config["triggers"]):
                return sentence.strip()
        return "Catalyst detected but specific trigger unclear"

    def _extract_idea_content(self, message: str, catalyst_type: str) -> str:
        """Extract the actual idea content from the message"""
        # Look for idea-introducing phrases
        idea_markers = ["what if", "maybe we could", "I think", "how about", "we could try"]
        
        for marker in idea_markers:
            if marker in message.lower():
                # Extract text after the marker
                marker_pos = message.lower().find(marker)
                if marker_pos != -1:
                    idea_content = message[marker_pos:].split('.')[0]
                    return idea_content.strip()
        
        # Fallback: return first sentence if it seems to contain an idea
        first_sentence = message.split('.')[0]
        if len(first_sentence) > 20:  # Minimum length for an idea
            return first_sentence.strip()
        
        return message[:100]  # Fallback

    def _generate_idea_title(self, idea_content: str) -> str:
        """Generate a concise title for the idea"""
        # Extract key concepts for title
        words = idea_content.split()
        # Remove common words and take first meaningful words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "what", "if", "we", "could", "maybe", "how", "about"}
        meaningful_words = [word for word in words if word.lower() not in stop_words]
        
        title = " ".join(meaningful_words[:5])  # First 5 meaningful words
        return title.title() if title else "New Idea"

    def _extract_catalyst_quote(self, message: str, catalyst_type: str) -> str:
        """Extract the specific quote that triggered the evolution"""
        # Return first sentence as the catalyst quote
        first_sentence = message.split('.')[0]
        return first_sentence.strip() if first_sentence else message[:50]

    def _calculate_novelty_score(self, idea_content: str, previous_ideas: List) -> float:
        """Calculate how novel this idea is compared to previous ones"""
        if not previous_ideas:
            return 1.0
        
        current_words = set(idea_content.lower().split())
        max_similarity = 0.0
        
        for prev_idea in previous_ideas:
            prev_words = set(prev_idea.current_form.lower().split())
            if prev_words:
                similarity = len(current_words.intersection(prev_words)) / len(current_words.union(prev_words))
                max_similarity = max(max_similarity, similarity)
        
        return 1.0 - max_similarity  # Higher novelty = lower similarity

    def _calculate_complexity_score(self, idea_content: str) -> float:
        """Calculate complexity score based on content analysis"""
        # Simple heuristics for complexity
        complex_words = ["system", "architecture", "framework", "implementation", "algorithm", "optimization", "integration"]
        simple_words = ["basic", "simple", "easy", "straightforward", "minimal", "quick"]
        
        complexity_indicators = sum(1 for word in complex_words if word in idea_content.lower())
        simplicity_indicators = sum(1 for word in simple_words if word in idea_content.lower())
        
        # Length-based complexity
        length_complexity = min(1.0, len(idea_content.split()) / 50)
        
        # Combine indicators
        word_complexity = (complexity_indicators - simplicity_indicators) / max(1, len(idea_content.split()))
        
        return max(0.0, min(1.0, length_complexity + word_complexity))

    def _calculate_feasibility_score(self, idea_content: str) -> float:
        """Calculate feasibility score based on content analysis"""
        # Heuristics for feasibility
        feasible_words = ["simple", "build", "create", "implement", "easy", "quick", "doable"]
        challenging_words = ["complex", "difficult", "impossible", "expensive", "time-consuming"]
        
        feasible_indicators = sum(1 for word in feasible_words if word in idea_content.lower())
        challenging_indicators = sum(1 for word in challenging_words if word in idea_content.lower())
        
        base_feasibility = 0.7  # Default moderate feasibility
        
        if feasible_indicators > challenging_indicators:
            return min(1.0, base_feasibility + 0.2)
        elif challenging_indicators > feasible_indicators:
            return max(0.1, base_feasibility - 0.3)
        
        return base_feasibility

    def _generate_test_suggestions(self, pattern: Dict) -> List[str]:
        """Generate specific test suggestions for a pattern hypothesis"""
        pattern_type = pattern["pattern_type"]
        
        test_suggestions = []
        
        if pattern_type == "ux_preference":
            test_suggestions = [
                "Present two solution options: one prioritizing simplicity, one prioritizing features",
                "Ask about past projects where UX decisions had to be made",
                "Explore preferences when there's a trade-off between ease-of-use and power"
            ]
        
        elif pattern_type == "technical_complexity":
            test_suggestions = [
                "Propose a technical solution and gauge comfort level",
                "Ask about learning preferences for new technologies",
                "Explore past experiences with complex vs simple tools"
            ]
        
        elif pattern_type == "problem_solving_style":
            test_suggestions = [
                "Present a problem and observe the approach taken",
                "Ask about preferred planning vs improvisation styles",
                "Explore how they handle unexpected challenges"
            ]
        
        elif pattern_type == "collaboration_style":
            test_suggestions = [
                "Discuss preferences for team vs solo work",
                "Ask about past collaboration experiences",
                "Explore communication style preferences"
            ]
        
        return test_suggestions

class CausalReasoningEngine:
    """Advanced causal reasoning and analogical thinking for project outcomes and cross-domain connections"""
    
    def __init__(self):
        # Causal relationship patterns
        self.causal_indicators = {
            "positive_outcomes": {
                "triggers": ["simplified", "streamlined", "reduced complexity", "made easier", "optimized", "improved", "enhanced"],
                "effects": ["increased", "improved", "better", "higher", "more", "boosted", "enhanced", "grew"],
                "metrics": ["engagement", "performance", "satisfaction", "retention", "conversion", "adoption", "usage"]
            },
            "negative_outcomes": {
                "triggers": ["complicated", "added complexity", "removed", "reduced", "eliminated", "cut"],
                "effects": ["decreased", "reduced", "lower", "worse", "dropped", "fell", "declined"],
                "metrics": ["engagement", "performance", "satisfaction", "retention", "conversion", "adoption", "usage"]
            },
            "implementation_patterns": {
                "interface_changes": ["UI", "interface", "design", "layout", "navigation", "user experience", "UX"],
                "feature_additions": ["added", "new feature", "functionality", "capability", "option", "tool"],
                "process_improvements": ["workflow", "process", "automation", "integration", "optimization"],
                "technical_changes": ["performance", "speed", "loading", "response time", "scalability", "architecture"]
            }
        }
        
        # Domain mapping for analogical thinking
        self.domain_analogies = {
            "mobile_app": {
                "related_domains": ["web_app", "desktop_app", "tablet_app"],
                "transferable_concepts": ["user_interface", "navigation", "performance", "user_engagement", "onboarding"],
                "common_challenges": ["small_screen", "touch_interface", "limited_attention", "performance_constraints"]
            },
            "web_app": {
                "related_domains": ["mobile_app", "desktop_app", "web_platform"],
                "transferable_concepts": ["responsive_design", "load_times", "browser_compatibility", "accessibility"],
                "common_challenges": ["cross_browser", "responsive_design", "SEO", "loading_speed"]
            },
            "e_commerce": {
                "related_domains": ["marketplace", "retail", "subscription_service"],
                "transferable_concepts": ["conversion_funnel", "user_journey", "payment_flow", "product_discovery"],
                "common_challenges": ["cart_abandonment", "checkout_friction", "product_findability", "trust_signals"]
            },
            "content_platform": {
                "related_domains": ["social_media", "blog", "news_site", "video_platform"],
                "transferable_concepts": ["content_discovery", "engagement_metrics", "recommendation_systems", "user_retention"],
                "common_challenges": ["content_quality", "user_engagement", "content_moderation", "recommendation_accuracy"]
            },
            "saas_product": {
                "related_domains": ["enterprise_software", "consumer_app", "platform_service"],
                "transferable_concepts": ["onboarding", "feature_adoption", "churn_reduction", "user_activation"],
                "common_challenges": ["feature_complexity", "user_onboarding", "pricing_strategy", "feature_discovery"]
            }
        }
        
        # Pattern templates for analogical reasoning
        self.analogy_templates = [
            "This challenge is similar to what we solved in {domain} when we {solution}",
            "Like in the {domain} project, the key insight here might be {insight}",
            "We faced a similar problem in {domain} and found that {approach} worked well",
            "Drawing from our {domain} experience, {strategy} could be effective here",
            "This reminds me of the {domain} optimization where {technique} yielded {outcome}"
        ]
        
        # Unintended consequence patterns
        self.consequence_patterns = {
            "interface_simplification": {
                "positive": ["easier_onboarding", "reduced_confusion", "faster_task_completion"],
                "potential_negative": ["feature_discoverability_loss", "power_user_frustration", "reduced_functionality_perception"]
            },
            "feature_addition": {
                "positive": ["increased_functionality", "user_satisfaction", "competitive_advantage"],
                "potential_negative": ["interface_complexity", "cognitive_overload", "maintenance_burden", "performance_impact"]
            },
            "performance_optimization": {
                "positive": ["faster_loading", "better_user_experience", "reduced_bounce_rate"],
                "potential_negative": ["increased_complexity", "technical_debt", "resource_intensive_maintenance"]
            },
            "automation_implementation": {
                "positive": ["efficiency_gains", "error_reduction", "consistency"],
                "potential_negative": ["job_displacement_concerns", "reduced_flexibility", "dependency_risks"]
            }
        }

    def extract_causal_relationships(self, text: str, user_id: int, db: Session) -> List[Dict]:
        """Extract cause-effect relationships from project descriptions and outcomes"""
        text_lower = text.lower()
        relationships = []
        
        # Look for explicit causal language patterns
        causal_patterns = [
            (r"when (?:we )?(.+?),\s*(.+?)(?:increased|decreased|improved|worsened|changed)(?:\s+by)?\s*(\d+(?:\.\d+)?%?)", "explicit_when"),
            (r"after (?:we )?(.+?),\s*(.+?)(?:went up|went down|improved|got better|got worse)", "explicit_after"),
            (r"(?:by|through) (.+?),\s*(?:we )?(?:achieved|saw|experienced|got)\s*(.+)", "explicit_by"),
            (r"(.+?)\s*(?:led to|resulted in|caused|brought about)\s*(.+)", "explicit_causation")
        ]
        
        import re
        for pattern, pattern_type in causal_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                if pattern_type == "explicit_when":
                    cause = match.group(1).strip()
                    effect = match.group(2).strip()
                    magnitude = match.group(3) if len(match.groups()) > 2 else None
                    
                    relationships.append({
                        "type": "causal_relationship",
                        "cause": cause,
                        "effect": effect,
                        "magnitude": magnitude,
                        "confidence": 0.9,
                        "pattern_type": pattern_type,
                        "domain": self._identify_domain(cause + " " + effect),
                        "transferability": self._assess_transferability(cause, effect)
                    })
        
        # Look for implicit causal relationships
        for outcome_type, patterns in self.causal_indicators.items():
            triggers_found = [t for t in patterns["triggers"] if t in text_lower]
            effects_found = [e for e in patterns["effects"] if e in text_lower]
            metrics_found = [m for m in patterns["metrics"] if m in text_lower]
            
            if triggers_found and effects_found and metrics_found:
                for trigger in triggers_found:
                    for effect in effects_found:
                        for metric in metrics_found:
                            relationships.append({
                                "type": "implicit_causal_relationship",
                                "cause": trigger,
                                "effect": f"{effect} {metric}",
                                "magnitude": None,
                                "confidence": 0.7,
                                "pattern_type": outcome_type,
                                "domain": self._identify_domain(text),
                                "transferability": self._assess_transferability(trigger, effect)
                            })
        
        return relationships

    def find_analogous_projects(self, current_challenge: str, user_id: int, db: Session) -> List[Dict]:
        """Find analogous projects and solutions from past experiences"""
        from database import Message, Conversation, Memory
        
        current_lower = current_challenge.lower()
        analogies = []
        
        # Get past conversations and memories
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc()).limit(50).all()
        
        memories = db.query(Memory).filter(
            Memory.user_id == user_id,
            Memory.importance_score > 0.4
        ).order_by(Memory.importance_score.desc()).limit(30).all()
        
        # Extract domain and challenge type from current text
        current_domain = self._identify_domain(current_challenge)
        current_challenge_type = self._identify_challenge_type(current_challenge)
        
        # Search for similar challenges in different domains
        for conv in conversations:
            messages = db.query(Message).filter(
                Message.conversation_id == conv.id,
                Message.role == "user"
            ).all()
            
            conv_content = " ".join([msg.content for msg in messages])
            conv_domain = self._identify_domain(conv_content)
            conv_challenge_type = self._identify_challenge_type(conv_content)
            
            # Check for analogical potential
            if (conv_domain != current_domain and 
                conv_challenge_type == current_challenge_type and
                self._has_solution_indicators(conv_content)):
                
                similarity_score = self._calculate_analogical_similarity(
                    current_challenge, conv_content, current_domain, conv_domain
                )
                
                if similarity_score > 0.6:
                    solution_extract = self._extract_solution_from_content(conv_content)
                    
                    analogies.append({
                        "type": "project_analogy",
                        "source_domain": conv_domain,
                        "target_domain": current_domain,
                        "challenge_type": conv_challenge_type,
                        "similarity_score": similarity_score,
                        "conversation_title": conv.title,
                        "solution_extract": solution_extract,
                        "analogy_insight": self._generate_analogy_insight(
                            current_domain, conv_domain, solution_extract
                        ),
                        "transferable_elements": self._identify_transferable_elements(
                            conv_content, current_domain
                        )
                    })
        
        # Check memories for analogical insights
        for memory in memories:
            memory_domain = self._identify_domain(memory.content)
            memory_challenge_type = self._identify_challenge_type(memory.content)
            
            if (memory_domain != current_domain and
                memory_challenge_type == current_challenge_type):
                
                similarity_score = self._calculate_analogical_similarity(
                    current_challenge, memory.content, current_domain, memory_domain
                )
                
                if similarity_score > 0.5:
                    analogies.append({
                        "type": "memory_analogy",
                        "source_domain": memory_domain,
                        "target_domain": current_domain,
                        "challenge_type": memory_challenge_type,
                        "similarity_score": similarity_score,
                        "memory_content": memory.content[:200],
                        "analogy_insight": self._generate_analogy_insight(
                            current_domain, memory_domain, memory.content
                        )
                    })
        
        # Sort by similarity and return top analogies
        analogies.sort(key=lambda x: x["similarity_score"], reverse=True)
        return analogies[:5]

    def predict_unintended_consequences(self, proposed_change: str, context: str) -> List[Dict]:
        """Identify potential unintended consequences before they occur"""
        change_lower = proposed_change.lower()
        context_lower = context.lower()
        
        predictions = []
        
        # Identify the type of change being proposed
        change_type = None
        for change_pattern, keywords in self.causal_indicators["implementation_patterns"].items():
            if any(keyword in change_lower for keyword in keywords):
                change_type = change_pattern
                break
        
        if not change_type:
            # Default categorization based on common patterns
            if any(word in change_lower for word in ["add", "new", "feature", "functionality"]):
                change_type = "feature_addition"
            elif any(word in change_lower for word in ["simplify", "reduce", "remove", "streamline"]):
                change_type = "interface_simplification"
            elif any(word in change_lower for word in ["optimize", "faster", "performance", "speed"]):
                change_type = "performance_optimization"
            elif any(word in change_lower for word in ["automate", "automatic", "ai", "machine"]):
                change_type = "automation_implementation"
        
        if change_type in self.consequence_patterns:
            consequences = self.consequence_patterns[change_type]
            
            # Predict positive outcomes
            for positive in consequences["positive"]:
                predictions.append({
                    "type": "positive_consequence",
                    "consequence": positive.replace("_", " "),
                    "probability": 0.7,
                    "impact": "medium",
                    "timeline": "short_term",
                    "mitigation": None
                })
            
            # Predict potential negative outcomes
            for negative in consequences["potential_negative"]:
                # Assess likelihood based on context
                likelihood = self._assess_consequence_likelihood(negative, context_lower)
                
                if likelihood > 0.3:
                    mitigation = self._suggest_mitigation_strategy(negative, change_type)
                    
                    predictions.append({
                        "type": "potential_negative_consequence",
                        "consequence": negative.replace("_", " "),
                        "probability": likelihood,
                        "impact": self._assess_impact_severity(negative),
                        "timeline": self._predict_timeline(negative),
                        "mitigation": mitigation,
                        "warning_level": "high" if likelihood > 0.7 else "medium" if likelihood > 0.5 else "low"
                    })
        
        return predictions

    def bridge_knowledge_domains(self, source_domain: str, target_domain: str, knowledge: str) -> Dict:
        """Bridge knowledge across different domains and contexts"""
        bridging_result = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "knowledge_extract": knowledge,
            "bridging_concepts": [],
            "adaptation_strategy": "",
            "transferability_score": 0.0,
            "potential_applications": []
        }
        
        # Find common concepts between domains
        if source_domain in self.domain_analogies and target_domain in self.domain_analogies:
            source_concepts = set(self.domain_analogies[source_domain]["transferable_concepts"])
            target_concepts = set(self.domain_analogies[target_domain]["transferable_concepts"])
            
            common_concepts = source_concepts.intersection(target_concepts)
            bridging_result["bridging_concepts"] = list(common_concepts)
            
            # Calculate transferability score
            transferability = len(common_concepts) / len(source_concepts.union(target_concepts))
            bridging_result["transferability_score"] = transferability
            
            # Generate adaptation strategy
            if transferability > 0.6:
                bridging_result["adaptation_strategy"] = f"High transferability - {source_domain} concepts can be directly applied to {target_domain} with minimal adaptation"
            elif transferability > 0.3:
                bridging_result["adaptation_strategy"] = f"Moderate transferability - {source_domain} concepts need contextual adaptation for {target_domain}"
            else:
                bridging_result["adaptation_strategy"] = f"Low transferability - {source_domain} concepts require significant reframing for {target_domain}"
            
            # Suggest potential applications
            for concept in common_concepts:
                application = self._generate_cross_domain_application(concept, source_domain, target_domain, knowledge)
                if application:
                    bridging_result["potential_applications"].append(application)
        
        return bridging_result

    def _identify_domain(self, text: str) -> str:
        """Identify the domain/project type from text content"""
        text_lower = text.lower()
        
        domain_keywords = {
            "mobile_app": ["mobile", "app", "ios", "android", "smartphone", "tablet"],
            "web_app": ["website", "web", "browser", "online", "url", "http"],
            "e_commerce": ["ecommerce", "shop", "store", "buy", "sell", "product", "cart", "checkout"],
            "content_platform": ["content", "blog", "article", "video", "media", "publish"],
            "saas_product": ["saas", "software", "platform", "service", "subscription", "enterprise"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
        
        return "general"

    def _identify_challenge_type(self, text: str) -> str:
        """Identify the type of challenge from text content"""
        text_lower = text.lower()
        
        challenge_types = {
            "user_engagement": ["engagement", "retention", "active users", "usage", "adoption"],
            "performance": ["slow", "performance", "speed", "loading", "response time"],
            "user_experience": ["ux", "user experience", "usability", "interface", "navigation"],
            "conversion": ["conversion", "signup", "purchase", "funnel", "abandon"],
            "feature_adoption": ["feature", "adoption", "discovery", "usage", "uptake"]
        }
        
        for challenge_type, keywords in challenge_types.items():
            if any(keyword in text_lower for keyword in keywords):
                return challenge_type
        
        return "general"

    def _has_solution_indicators(self, text: str) -> bool:
        """Check if text contains solution or outcome indicators"""
        solution_indicators = [
            "solved", "fixed", "improved", "increased", "decreased", "optimized",
            "implemented", "changed", "modified", "updated", "redesigned",
            "result", "outcome", "effect", "impact", "success"
        ]
        return any(indicator in text.lower() for indicator in solution_indicators)

    def _calculate_analogical_similarity(self, text1: str, text2: str, domain1: str, domain2: str) -> float:
        """Calculate similarity score for analogical reasoning"""
        # Basic word overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        # Remove common words
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        meaningful_words1 = words1 - common_words
        meaningful_words2 = words2 - common_words
        
        if not meaningful_words1 or not meaningful_words2:
            return 0.0
        
        word_overlap = len(meaningful_words1.intersection(meaningful_words2))
        word_similarity = word_overlap / len(meaningful_words1.union(meaningful_words2))
        
        # Domain similarity bonus
        domain_similarity = 0.0
        if domain1 in self.domain_analogies and domain2 in self.domain_analogies:
            if domain2 in self.domain_analogies[domain1]["related_domains"]:
                domain_similarity = 0.3
        
        return word_similarity + domain_similarity

    def _extract_solution_from_content(self, content: str) -> str:
        """Extract solution description from conversation content"""
        sentences = content.split('.')
        solution_sentences = []
        
        solution_keywords = ["solved", "fixed", "implemented", "changed", "improved", "optimized", "redesigned"]
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in solution_keywords):
                solution_sentences.append(sentence.strip())
        
        return ". ".join(solution_sentences[:2]) if solution_sentences else "Solution details not clearly extractable"

    def _generate_analogy_insight(self, target_domain: str, source_domain: str, solution: str) -> str:
        """Generate analogical insight statement"""
        template = self.analogy_templates[hash(source_domain + target_domain) % len(self.analogy_templates)]
        
        insight = template.format(
            domain=source_domain.replace("_", " "),
            solution=solution[:100] + "..." if len(solution) > 100 else solution,
            insight="similar underlying patterns",
            approach="the same strategy",
            strategy="a parallel approach",
            technique="similar optimization",
            outcome="positive results"
        )
        
        return insight

    def _identify_transferable_elements(self, content: str, target_domain: str) -> List[str]:
        """Identify elements that can be transferred between domains"""
        transferable = []
        
        if target_domain in self.domain_analogies:
            concepts = self.domain_analogies[target_domain]["transferable_concepts"]
            
            for concept in concepts:
                if concept.replace("_", " ") in content.lower():
                    transferable.append(concept.replace("_", " "))
        
        return transferable

    def _assess_consequence_likelihood(self, consequence: str, context: str) -> float:
        """Assess likelihood of a consequence occurring"""
        base_likelihood = 0.5
        
        # Context-based adjustments
        if "complex" in context or "advanced" in context:
            if "complexity" in consequence:
                base_likelihood += 0.2
        
        if "user" in context and "new" in context:
            if "confusion" in consequence or "discoverability" in consequence:
                base_likelihood += 0.3
        
        if "performance" in context and "optimization" in consequence:
            if "maintenance" in consequence:
                base_likelihood += 0.2
        
        return min(1.0, base_likelihood)

    def _assess_impact_severity(self, consequence: str) -> str:
        """Assess the potential impact severity of a consequence"""
        high_impact_keywords = ["frustration", "loss", "abandonment", "failure", "critical"]
        medium_impact_keywords = ["confusion", "difficulty", "reduced", "slower"]
        
        if any(keyword in consequence for keyword in high_impact_keywords):
            return "high"
        elif any(keyword in consequence for keyword in medium_impact_keywords):
            return "medium"
        else:
            return "low"

    def _predict_timeline(self, consequence: str) -> str:
        """Predict when a consequence might manifest"""
        immediate_keywords = ["confusion", "frustration", "difficulty"]
        medium_term_keywords = ["adoption", "discovery", "learning"]
        long_term_keywords = ["maintenance", "debt", "scalability"]
        
        if any(keyword in consequence for keyword in immediate_keywords):
            return "immediate"
        elif any(keyword in consequence for keyword in medium_term_keywords):
            return "medium_term"
        else:
            return "long_term"

    def _suggest_mitigation_strategy(self, consequence: str, change_type: str) -> str:
        """Suggest mitigation strategies for potential negative consequences"""
        mitigation_strategies = {
            "feature_discoverability_loss": "Implement progressive disclosure or onboarding tooltips",
            "power_user_frustration": "Provide advanced/expert mode toggle",
            "interface_complexity": "Use progressive disclosure and smart defaults",
            "cognitive_overload": "Implement contextual help and guided tours",
            "maintenance_burden": "Establish automated testing and documentation standards",
            "performance_impact": "Implement performance monitoring and optimization cycles",
            "dependency_risks": "Create fallback mechanisms and redundancy plans"
        }
        
        consequence_key = consequence.replace(" ", "_")
        return mitigation_strategies.get(consequence_key, "Monitor closely and gather user feedback")

    def _generate_cross_domain_application(self, concept: str, source_domain: str, target_domain: str, knowledge: str) -> str:
        """Generate specific application for cross-domain knowledge transfer"""
        applications = {
            "user_interface": f"Apply {source_domain} interface patterns to improve {target_domain} usability",
            "performance": f"Transfer {source_domain} optimization techniques to enhance {target_domain} speed",
            "user_engagement": f"Adapt {source_domain} engagement strategies for {target_domain} retention",
            "onboarding": f"Implement {source_domain} onboarding flow in {target_domain} context"
        }
        
        return applications.get(concept, f"Apply {concept} insights from {source_domain} to {target_domain}")

class ConceptHierarchy:
    """Manages hierarchical concept relationships and semantic clustering"""
    
    def __init__(self):
        # Abstract concept categories for semantic clustering
        self.concept_hierarchies = {
            "creativity": {
                "parent_concepts": ["innovation", "imagination", "originality"],
                "child_concepts": ["brainstorming", "ideation", "artistic_expression", "problem_solving"],
                "related_domains": ["art", "science", "technology", "design"],
                "principles": ["biomimicry", "synthesis", "divergent_thinking", "iteration"]
            },
            "systems_thinking": {
                "parent_concepts": ["complexity", "emergence", "interconnection"],
                "child_concepts": ["feedback_loops", "patterns", "structures", "dynamics"],
                "related_domains": ["biology", "engineering", "ecology", "business"],
                "principles": ["holism", "non_linearity", "adaptation", "resilience"]
            },
            "learning": {
                "parent_concepts": ["knowledge", "understanding", "growth"],
                "child_concepts": ["practice", "repetition", "reflection", "application"],
                "related_domains": ["education", "psychology", "neuroscience", "pedagogy"],
                "principles": ["spaced_repetition", "active_recall", "metacognition", "transfer"]
            },
            "communication": {
                "parent_concepts": ["connection", "expression", "understanding"],
                "child_concepts": ["listening", "speaking", "writing", "body_language"],
                "related_domains": ["linguistics", "psychology", "media", "technology"],
                "principles": ["clarity", "empathy", "context", "feedback"]
            },
            "problem_solving": {
                "parent_concepts": ["analysis", "synthesis", "evaluation"],
                "child_concepts": ["decomposition", "pattern_recognition", "hypothesis", "testing"],
                "related_domains": ["mathematics", "science", "engineering", "philosophy"],
                "principles": ["first_principles", "analogical_reasoning", "constraint_satisfaction", "optimization"]
            },
            "adaptation": {
                "parent_concepts": ["flexibility", "resilience", "evolution"],
                "child_concepts": ["adjustment", "learning", "survival", "optimization"],
                "related_domains": ["biology", "psychology", "business", "technology"],
                "principles": ["feedback", "variation", "selection", "inheritance"]
            }
        }
        
        # Semantic similarity clusters for concept grouping
        self.semantic_clusters = {
            "biological_systems": ["ecosystem", "organism", "cell", "DNA", "evolution", "adaptation", "symbiosis", "metabolism"],
            "natural_patterns": ["fibonacci", "fractals", "spirals", "waves", "cycles", "growth", "decay", "emergence"],
            "design_principles": ["form_follows_function", "minimalism", "modularity", "scalability", "usability", "aesthetics"],
            "cognitive_processes": ["memory", "attention", "perception", "reasoning", "intuition", "creativity", "consciousness"],
            "network_effects": ["connections", "nodes", "hubs", "clusters", "viral", "cascade", "emergence", "collective"],
            "flow_states": ["focus", "immersion", "optimal_experience", "challenge", "skill", "feedback", "clear_goals"],
            "optimization": ["efficiency", "effectiveness", "minimal", "maximal", "trade_offs", "constraints", "objectives"]
        }
        
        # Cross-domain bridges for unexpected connections
        self.domain_bridges = {
            ("biology", "technology"): ["biomimicry", "neural_networks", "genetic_algorithms", "swarm_intelligence"],
            ("psychology", "design"): ["user_experience", "cognitive_load", "mental_models", "behavioral_economics"],
            ("physics", "philosophy"): ["emergence", "determinism", "quantum_mechanics", "observer_effect"],
            ("music", "mathematics"): ["harmony", "rhythm", "patterns", "frequencies", "golden_ratio"],
            ("nature", "architecture"): ["organic_forms", "structural_efficiency", "environmental_adaptation", "sustainable_design"],
            ("games", "learning"): ["gamification", "progression", "rewards", "challenge", "engagement", "mastery"]
        }
        
        # Temporal concept activation for surfacing dormant ideas
        self.concept_activation_history = {}  # Track when concepts were last activated
        self.concept_relevance_decay = 30  # Days before concepts become "dormant"

    def extract_concepts(self, text: str) -> List[Dict]:
        """Extract semantic concepts from text using hierarchical analysis"""
        text_lower = text.lower()
        extracted_concepts = []
        
        # Check for direct concept matches in hierarchies
        for category, hierarchy in self.concept_hierarchies.items():
            relevance_score = 0
            matched_elements = []
            
            # Check parent concepts
            for concept in hierarchy["parent_concepts"]:
                if any(word in text_lower for word in concept.split("_")):
                    relevance_score += 3
                    matched_elements.append(f"parent:{concept}")
            
            # Check child concepts
            for concept in hierarchy["child_concepts"]:
                if any(word in text_lower for word in concept.split("_")):
                    relevance_score += 2
                    matched_elements.append(f"child:{concept}")
            
            # Check related domains
            for domain in hierarchy["related_domains"]:
                if domain in text_lower:
                    relevance_score += 1
                    matched_elements.append(f"domain:{domain}")
            
            # Check principles
            for principle in hierarchy["principles"]:
                if any(word in text_lower for word in principle.split("_")):
                    relevance_score += 2
                    matched_elements.append(f"principle:{principle}")
            
            if relevance_score > 0:
                extracted_concepts.append({
                    "category": category,
                    "relevance_score": relevance_score,
                    "matched_elements": matched_elements,
                    "activation_strength": min(1.0, relevance_score / 5)
                })
        
        # Check semantic clusters
        for cluster_name, concepts in self.semantic_clusters.items():
            cluster_score = 0
            matched_concepts = []
            
            for concept in concepts:
                if any(word in text_lower for word in concept.split("_")):
                    cluster_score += 1
                    matched_concepts.append(concept)
            
            if cluster_score > 0:
                extracted_concepts.append({
                    "category": f"cluster:{cluster_name}",
                    "relevance_score": cluster_score,
                    "matched_elements": matched_concepts,
                    "activation_strength": min(1.0, cluster_score / 3)
                })
        
        return extracted_concepts

    def find_cross_domain_connections(self, current_concepts: List[str], memory_concepts: List[str]) -> List[Dict]:
        """Find unexpected connections across different domains"""
        connections = []
        
        # Check for domain bridges
        for (domain1, domain2), bridge_concepts in self.domain_bridges.items():
            current_in_domain1 = any(d1_word in " ".join(current_concepts).lower() for d1_word in domain1.split("_"))
            memory_in_domain2 = any(d2_word in " ".join(memory_concepts).lower() for d2_word in domain2.split("_"))
            
            if current_in_domain1 and memory_in_domain2:
                # Find applicable bridge concepts
                applicable_bridges = []
                combined_concepts = current_concepts + memory_concepts
                
                for bridge in bridge_concepts:
                    bridge_words = bridge.split("_")
                    if any(word in " ".join(combined_concepts).lower() for word in bridge_words):
                        applicable_bridges.append(bridge)
                
                if applicable_bridges:
                    connections.append({
                        "type": "cross_domain_bridge",
                        "domains": (domain1, domain2),
                        "bridge_concepts": applicable_bridges,
                        "connection_strength": len(applicable_bridges) / len(bridge_concepts),
                        "insight": f"Connection between {domain1} and {domain2} through {', '.join(applicable_bridges)}"
                    })
        
        return connections

    def surface_dormant_concepts(self, current_concepts: List[str], user_id: int, db: Session) -> List[Dict]:
        """Surface dormant concepts that become relevant to current context"""
        from database import Memory
        
        dormant_concepts = []
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=self.concept_relevance_decay)
        
        # Get old memories that haven't been accessed recently
        old_memories = db.query(Memory).filter(
            Memory.user_id == user_id,
            Memory.last_accessed < cutoff_date,
            Memory.importance_score > 0.4
        ).order_by(Memory.created_at.asc()).limit(20).all()
        
        for memory in old_memories:
            memory_concepts = self.extract_concepts(memory.content)
            
            if memory_concepts:
                # Check for semantic similarity with current concepts
                for mem_concept in memory_concepts:
                    for curr_concept in current_concepts:
                        # Find semantic overlaps
                        similarity_score = self._calculate_semantic_similarity(
                            mem_concept.get("category", ""), curr_concept
                        )
                        
                        if similarity_score > 0.3:  # Threshold for relevance
                            days_dormant = (datetime.now(timezone.utc) - memory.last_accessed).days
                            
                            dormant_concepts.append({
                                "memory_id": memory.id,
                                "memory_content": memory.content[:200],
                                "concept_category": mem_concept.get("category"),
                                "similarity_score": similarity_score,
                                "days_dormant": days_dormant,
                                "reactivation_reason": f"Connects to current {curr_concept} through {mem_concept.get('category')}",
                                "creative_potential": similarity_score * (1 + days_dormant / 100)  # Older = more creative potential
                            })
        
        # Sort by creative potential and return top 5
        dormant_concepts.sort(key=lambda x: x["creative_potential"], reverse=True)
        return dormant_concepts[:5]

    def _calculate_semantic_similarity(self, concept1: str, concept2: str) -> float:
        """Calculate semantic similarity between concepts"""
        # Check if concepts are in the same hierarchy
        for category, hierarchy in self.concept_hierarchies.items():
            all_concepts = (hierarchy["parent_concepts"] + hierarchy["child_concepts"] + 
                          hierarchy["related_domains"] + hierarchy["principles"])
            
            if concept1 in all_concepts or concept2 in all_concepts:
                if concept1 == concept2:
                    return 1.0
                elif concept1 in all_concepts and concept2 in all_concepts:
                    return 0.8
                else:
                    return 0.4
        
        # Check semantic clusters
        for cluster_concepts in self.semantic_clusters.values():
            if concept1 in cluster_concepts and concept2 in cluster_concepts:
                return 0.7
        
        # Basic word overlap
        words1 = set(concept1.lower().split("_"))
        words2 = set(concept2.lower().split("_"))
        if words1.intersection(words2):
            return 0.3
        
        return 0.0

    def generate_creative_combinations(self, active_concepts: List[Dict], dormant_concepts: List[Dict]) -> List[Dict]:
        """Generate novel creative combinations from active and dormant concepts"""
        combinations = []
        
        for active in active_concepts:
            for dormant in dormant_concepts:
                # Look for unexpected combinations with high creative potential
                combination_strength = (active.get("activation_strength", 0) + 
                                      dormant.get("creative_potential", 0)) / 2
                
                if combination_strength > 0.5:
                    combinations.append({
                        "active_concept": active.get("category"),
                        "dormant_concept": dormant.get("concept_category"),
                        "combination_strength": combination_strength,
                        "creative_insight": self._generate_creative_insight(active, dormant),
                        "application_domains": self._suggest_application_domains(active, dormant)
                    })
        
        return sorted(combinations, key=lambda x: x["combination_strength"], reverse=True)[:3]

    def _generate_creative_insight(self, active_concept: Dict, dormant_concept: Dict) -> str:
        """Generate creative insight from concept combination"""
        insights = [
            f"What if we applied {dormant_concept.get('concept_category', 'past insights')} to {active_concept.get('category', 'current context')}?",
            f"The principles from {dormant_concept.get('concept_category', 'previous work')} might unlock new approaches to {active_concept.get('category', 'this challenge')}",
            f"Consider how {dormant_concept.get('concept_category', 'that old idea')} could transform your thinking about {active_concept.get('category', 'current focus')}",
            f"There's an unexplored connection between {active_concept.get('category', 'what you're working on')} and {dormant_concept.get('concept_category', 'past discoveries')}"
        ]
        return insights[hash(str(active_concept) + str(dormant_concept)) % len(insights)]

    def _suggest_application_domains(self, active_concept: Dict, dormant_concept: Dict) -> List[str]:
        """Suggest domains where concept combination could be applied"""
        # Extract domains from both concepts
        active_category = active_concept.get("category", "")
        dormant_category = dormant_concept.get("concept_category", "")
        
        domains = []
        
        # Get domains from hierarchies
        for category, hierarchy in self.concept_hierarchies.items():
            if category in active_category or category in dormant_category:
                domains.extend(hierarchy["related_domains"])
        
        # Add cross-domain applications
        if len(domains) >= 2:
            for (d1, d2), bridges in self.domain_bridges.items():
                if d1 in domains or d2 in domains:
                    domains.extend([d1, d2])
        
        return list(set(domains))[:4]

class CognitiveProcessor:
    """Advanced cognitive processing for the AI agent"""

    def __init__(self):
        # Initialize concept hierarchy system
        self.concept_hierarchy = ConceptHierarchy()
        
        # Enhanced associative recall patterns
        self.recall_triggers = {
            "routine": ["morning", "wake up", "get up", "start day", "breakfast", "coffee", "schedule"],
            "struggle": ["difficult", "hard", "challenge", "problem", "issue", "trouble", "stuck", "frustrated"],
            "emotions": ["feel", "feeling", "emotion", "mood", "angry", "sad", "happy", "anxious", "stressed"],
            "habits": ["habit", "routine", "daily", "regularly", "always", "usually", "often", "tend to"],
            "work": ["job", "work", "office", "meeting", "deadline", "project", "boss", "colleague"],
            "health": ["tired", "energy", "sleep", "exercise", "diet", "sick", "doctor", "pain"],
            "relationships": ["partner", "friend", "family", "parent", "child", "spouse", "date", "social"],
            "learning": ["learn", "study", "read", "practice", "skill", "course", "education", "growth"],
            "goals": ["goal", "plan", "want to", "hoping", "trying", "working on", "achieve", "accomplish"],
            "time": ["time", "schedule", "busy", "free", "calendar", "appointment", "deadline", "urgent"]
        }
        
        # Context-sensitive response patterns
        self.response_adaptations = {
            "struggling": {
                "tone": "supportive and gentle",
                "approach": "offer understanding and practical help",
                "language": "reassuring and non-judgmental"
            },
            "excited": {
                "tone": "enthusiastic and encouraging",
                "approach": "match their energy and explore the positive momentum",
                "language": "celebratory and forward-looking"
            },
            "confused": {
                "tone": "patient and clarifying",
                "approach": "break things down and provide clear guidance",
                "language": "simple and step-by-step"
            },
            "tired": {
                "tone": "understanding and gentle",
                "approach": "focus on rest and self-care",
                "language": "calm and soothing"
            },
            "motivated": {
                "tone": "energetic and action-oriented",
                "approach": "help channel motivation into concrete steps",
                "language": "empowering and goal-focused"
            }
        }
        
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

    def find_associative_memories(self, current_message: str, user_id: int, db: Session) -> List[Dict]:
        """Find relevant memories through associative recall and semantic clustering"""
        from database import Memory, Message, Conversation
        
        current_lower = current_message.lower()
        relevant_memories = []
        
        # Extract concepts from current message using hierarchy system
        current_concepts = self.concept_hierarchy.extract_concepts(current_message)
        current_concept_names = [c.get("category", "") for c in current_concepts]
        
        # Find triggered recall categories (traditional method)
        triggered_categories = []
        for category, keywords in self.recall_triggers.items():
            if any(keyword in current_lower for keyword in keywords):
                triggered_categories.append(category)
        
        # Search memories for both traditional and semantic relevance
        memories = db.query(Memory).filter(
            Memory.user_id == user_id,
            Memory.importance_score > 0.3
        ).order_by(Memory.importance_score.desc()).limit(30).all()
        
        for memory in memories:
            memory_content = memory.content.lower()
            relevance_score = 0
            semantic_score = 0
            
            # Traditional keyword matching
            for category in triggered_categories:
                category_keywords = self.recall_triggers[category]
                matches = sum(1 for keyword in category_keywords if keyword in memory_content)
                if matches > 0:
                    relevance_score += matches / len(category_keywords)
            
            # Semantic concept matching
            memory_concepts = self.concept_hierarchy.extract_concepts(memory.content)
            memory_concept_names = [c.get("category", "") for c in memory_concepts]
            
            # Calculate semantic similarity
            for current_concept in current_concept_names:
                for memory_concept in memory_concept_names:
                    similarity = self.concept_hierarchy._calculate_semantic_similarity(current_concept, memory_concept)
                    semantic_score += similarity
            
            # Normalize semantic score
            if memory_concepts and current_concepts:
                semantic_score = semantic_score / (len(memory_concepts) * len(current_concepts))
            
            # Combined relevance score
            combined_score = relevance_score * 0.6 + semantic_score * 0.4
            
            if combined_score > 0.15:  # Lower threshold to catch semantic matches
                relevant_memories.append({
                    "content": memory.content,
                    "category": memory.category,
                    "relevance_score": relevance_score,
                    "semantic_score": semantic_score,
                    "combined_score": combined_score,
                    "created_at": memory.created_at,
                    "importance": memory.importance_score,
                    "memory_concepts": memory_concept_names,
                    "concept_overlap": len(set(current_concept_names).intersection(set(memory_concept_names)))
                })
        
        # Surface dormant concepts
        dormant_concepts = self.concept_hierarchy.surface_dormant_concepts(
            current_concept_names, user_id, db
        )
        
        # Add dormant concepts as special memory entries
        for dormant in dormant_concepts:
            relevant_memories.append({
                "content": f"💡 DORMANT INSIGHT: {dormant['memory_content']}",
                "category": "dormant_concept",
                "relevance_score": 0,
                "semantic_score": dormant["similarity_score"],
                "combined_score": dormant["creative_potential"],
                "created_at": datetime.now(timezone.utc) - timedelta(days=dormant["days_dormant"]),
                "importance": dormant["creative_potential"],
                "memory_concepts": [dormant["concept_category"]],
                "concept_overlap": 1,
                "dormant_info": dormant
            })
        
        # Sort by combined relevance and return top 5
        relevant_memories.sort(key=lambda x: x["combined_score"] * x["importance"], reverse=True)
        return relevant_memories[:5]
    
    def find_related_conversations(self, current_message: str, user_id: int, db: Session) -> List[Dict]:
        """Find past conversations with similar themes"""
        from database import Message, Conversation
        
        current_lower = current_message.lower()
        
        # Get recent conversations (last 30 days)
        thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
        conversations = db.query(Conversation).filter(
            Conversation.user_id == user_id,
            Conversation.updated_at > thirty_days_ago
        ).order_by(Conversation.updated_at.desc()).limit(10).all()
        
        related_convs = []
        
        for conv in conversations:
            # Get messages from this conversation
            messages = db.query(Message).filter(
                Message.conversation_id == conv.id,
                Message.role == "user"  # Focus on user messages
            ).limit(5).all()
            
            # Calculate thematic similarity
            similarity_score = 0
            message_contents = [msg.content.lower() for msg in messages]
            combined_content = " ".join(message_contents)
            
            # Check for keyword overlap
            current_words = set(current_lower.split())
            conv_words = set(combined_content.split())
            
            # Find meaningful word overlap (exclude common words)
            common_words = ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"]
            meaningful_current = current_words - set(common_words)
            meaningful_conv = conv_words - set(common_words)
            
            if meaningful_current and meaningful_conv:
                overlap = len(meaningful_current.intersection(meaningful_conv))
                similarity_score = overlap / max(len(meaningful_current), len(meaningful_conv))
            
            if similarity_score > 0.2:  # Threshold for relevance
                related_convs.append({
                    "conversation_id": conv.id,
                    "title": conv.title,
                    "similarity_score": similarity_score,
                    "updated_at": conv.updated_at,
                    "sample_messages": message_contents[:2]  # First 2 user messages
                })
        
        # Sort by similarity and return top 2
        related_convs.sort(key=lambda x: x["similarity_score"], reverse=True)
        return related_convs[:2]
    
    def detect_user_state(self, message: str, user_id: int, db: Session) -> Dict:
        """Detect user's current emotional and contextual state"""
        message_lower = message.lower()
        
        # Detect emotional indicators
        emotional_state = "neutral"
        emotional_intensity = 0.5
        
        struggle_indicators = ["difficult", "hard", "struggling", "stuck", "frustrated", "can't", "failed", "problem"]
        excitement_indicators = ["excited", "great", "amazing", "love", "wonderful", "fantastic", "awesome"]
        confusion_indicators = ["confused", "don't understand", "not sure", "unclear", "lost", "help"]
        tired_indicators = ["tired", "exhausted", "worn out", "drained", "no energy", "sleepy"]
        motivated_indicators = ["motivated", "ready", "determined", "focused", "goal", "achieve", "want to"]
        
        if any(indicator in message_lower for indicator in struggle_indicators):
            emotional_state = "struggling"
            emotional_intensity = 0.7
        elif any(indicator in message_lower for indicator in excitement_indicators):
            emotional_state = "excited"
            emotional_intensity = 0.8
        elif any(indicator in message_lower for indicator in confusion_indicators):
            emotional_state = "confused"
            emotional_intensity = 0.6
        elif any(indicator in message_lower for indicator in tired_indicators):
            emotional_state = "tired"
            emotional_intensity = 0.4
        elif any(indicator in message_lower for indicator in motivated_indicators):
            emotional_state = "motivated"
            emotional_intensity = 0.8
        
        # Detect time context
        time_context = "general"
        if any(word in message_lower for word in ["morning", "wake up", "start day"]):
            time_context = "morning"
        elif any(word in message_lower for word in ["evening", "night", "tired", "end of day"]):
            time_context = "evening"
        elif any(word in message_lower for word in ["work", "office", "meeting", "deadline"]):
            time_context = "work"
        
        return {
            "emotional_state": emotional_state,
            "emotional_intensity": emotional_intensity,
            "time_context": time_context,
            "message_length": len(message.split()),
            "question_count": message.count("?")
        }
    
    def generate_contextual_response_guidance(self, user_state: Dict, relevant_memories: List[Dict], related_conversations: List[Dict], current_message: str = "") -> str:
        """Generate guidance for contextual response adaptation with semantic insights"""
        guidance_parts = []
        
        # Add emotional state guidance
        emotional_state = user_state.get("emotional_state", "neutral")
        if emotional_state in self.response_adaptations:
            adaptation = self.response_adaptations[emotional_state]
            guidance_parts.append(f"User appears to be {emotional_state}. Adopt a {adaptation['tone']} tone. {adaptation['approach']}. Use {adaptation['language']} language.")
        
        # Extract current concepts for semantic analysis
        if current_message:
            current_concepts = self.concept_hierarchy.extract_concepts(current_message)
            
            if current_concepts:
                concept_guidance = "ACTIVE CONCEPTS:\n"
                for concept in current_concepts[:3]:  # Top 3 concepts
                    concept_guidance += f"- {concept['category']} (strength: {concept['activation_strength']:.2f})\n"
                guidance_parts.append(concept_guidance)
        
        # Add semantic memory analysis
        if relevant_memories:
            memory_context = "SEMANTIC MEMORY ANALYSIS:\n"
            dormant_insights = []
            regular_memories = []
            
            for memory in relevant_memories:
                if memory.get('category') == 'dormant_concept':
                    dormant_insights.append(memory)
                else:
                    regular_memories.append(memory)
            
            # Regular memories with semantic scores
            if regular_memories:
                memory_context += "Relevant memories:\n"
                for memory in regular_memories[:3]:
                    semantic_indicator = "🧠" if memory.get('semantic_score', 0) > 0.3 else "📝"
                    concepts = ", ".join(memory.get('memory_concepts', [])[:2])
                    memory_context += f"{semantic_indicator} {memory['content'][:80]}{'...' if len(memory['content']) > 80 else ''}"
                    if concepts:
                        memory_context += f" [Concepts: {concepts}]"
                    memory_context += "\n"
            
            # Dormant insights
            if dormant_insights:
                memory_context += "\n💡 DORMANT INSIGHTS (Creative potential):\n"
                for insight in dormant_insights[:2]:
                    dormant_info = insight.get('dormant_info', {})
                    memory_context += f"- {dormant_info.get('reactivation_reason', 'Unexpected connection')}\n"
                    memory_context += f"  Memory: {insight['content'][20:100]}{'...' if len(insight['content']) > 100 else ''}\n"
            
            guidance_parts.append(memory_context)
        
        # Add cross-domain connections if available
        if current_message and relevant_memories:
            current_concept_names = [c.get("category", "") for c in self.concept_hierarchy.extract_concepts(current_message)]
            memory_concept_names = []
            for memory in relevant_memories:
                memory_concept_names.extend(memory.get('memory_concepts', []))
            
            cross_domain_connections = self.concept_hierarchy.find_cross_domain_connections(
                current_concept_names, memory_concept_names
            )
            
            if cross_domain_connections:
                connection_guidance = "🌉 CROSS-DOMAIN BRIDGES:\n"
                for connection in cross_domain_connections[:2]:
                    connection_guidance += f"- {connection['insight']}\n"
                    connection_guidance += f"  Bridge concepts: {', '.join(connection['bridge_concepts'])}\n"
                guidance_parts.append(connection_guidance)
        
        # Add conversation patterns
        if related_conversations:
            pattern_context = "Similar past discussions:\n"
            for conv in related_conversations:
                pattern_context += f"- {conv['title']}: {', '.join(conv['sample_messages'][:1])}\n"
            guidance_parts.append(pattern_context)
        
        # Add time-sensitive guidance
        time_context = user_state.get("time_context", "general")
        if time_context == "morning":
            guidance_parts.append("Consider morning-specific context: energy levels, routine, starting the day.")
        elif time_context == "evening":
            guidance_parts.append("Consider evening context: fatigue, reflection, winding down.")
        elif time_context == "work":
            guidance_parts.append("Consider work context: professional challenges, time pressure, productivity.")
        
        return "\n\n".join(guidance_parts) if guidance_parts else ""

    def get_personality_adapted_prompt(self, user_id: int, base_prompt: str, db: Session, current_message: str = "") -> str:
        """Adapt system prompt based on learned user preferences, personality, and current context"""
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

        # Add associative recall and context-sensitive adaptations
        if current_message:
            user_state = self.detect_user_state(current_message, user_id, db)
            relevant_memories = self.find_associative_memories(current_message, user_id, db)
            related_conversations = self.find_related_conversations(current_message, user_id, db)
            
            contextual_guidance = self.generate_contextual_response_guidance(
                user_state, relevant_memories, related_conversations, current_message
            )
            
            if contextual_guidance:
                adaptations.append(f"CONTEXTUAL GUIDANCE:\n{contextual_guidance}")
            
            # Add creative combination insights
            current_concepts = self.concept_hierarchy.extract_concepts(current_message)
            dormant_concepts = self.concept_hierarchy.surface_dormant_concepts(
                [c.get("category", "") for c in current_concepts], user_id, db
            )
            
            if current_concepts and dormant_concepts:
                creative_combinations = self.concept_hierarchy.generate_creative_combinations(
                    current_concepts, dormant_concepts
                )
                
                if creative_combinations:
                    creativity_guidance = "🎨 CREATIVE SYNTHESIS OPPORTUNITIES:\n"
                    for combo in creative_combinations[:2]:
                        creativity_guidance += f"- {combo['creative_insight']}\n"
                        if combo['application_domains']:
                            creativity_guidance += f"  Applications: {', '.join(combo['application_domains'][:3])}\n"
                    adaptations.append(creativity_guidance)

        if adaptations:
            adaptation_text = "\n\nPersonality & Context Adaptations:\n" + "\n".join([f"- {adapt}" for adapt in adaptations])
            return base_prompt + adaptation_text

        return base_prompt

# Global cognitive processor, emotional tracker, contradiction detector, causal reasoning engine, and hypothesis formation engine
cognitive_processor = CognitiveProcessor()
emotional_tracker = EmotionalTrajectoryTracker()
contradiction_detector = ContradictionDetector()
causal_reasoning_engine = CausalReasoningEngine()
hypothesis_formation_engine = HypothesisFormationEngine()

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

@app.post("/api/associative-recall/test")
async def test_associative_recall(
    message: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test associative recall functionality"""
    user_state = cognitive_processor.detect_user_state(message, current_user.id, db)
    relevant_memories = cognitive_processor.find_associative_memories(message, current_user.id, db)
    related_conversations = cognitive_processor.find_related_conversations(message, current_user.id, db)
    
    contextual_guidance = cognitive_processor.generate_contextual_response_guidance(
        user_state, relevant_memories, related_conversations
    )
    
    return {
        "message": message,
        "user_state": user_state,
        "relevant_memories": relevant_memories,
        "related_conversations": related_conversations,
        "contextual_guidance": contextual_guidance,
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

# Mental Health Tracking Endpoints
@app.get("/api/mental-health/trajectory")
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_emotional_trajectory(
    days_back: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's emotional trajectory over time"""
    trajectory = emotional_tracker.analyze_emotional_trajectory(current_user.id, db, days_back)
    return {
        "trajectory": trajectory,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/mental-health/patterns")
@cache_response(ttl=60)  # Cache for 1 minute
async def get_emotional_patterns(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get detected emotional patterns"""
    patterns = emotional_tracker.detect_emotional_patterns(current_user.id, db)
    return {
        "patterns": patterns,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/mental-health/insights")
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_mental_health_insights(
    unacknowledged_only: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get mental health insights and recommendations"""
    from database import MentalHealthInsight
    
    query = db.query(MentalHealthInsight).filter(
        MentalHealthInsight.user_id == current_user.id
    )
    
    if unacknowledged_only:
        query = query.filter(MentalHealthInsight.is_acknowledged == False)
    
    insights = query.order_by(MentalHealthInsight.created_at.desc()).limit(20).all()
    
    result = []
    for insight in insights:
        result.append({
            "id": insight.id,
            "insight_type": insight.insight_type,
            "title": insight.title,
            "description": insight.description,
            "severity_level": insight.severity_level,
            "suggested_actions": insight.suggested_actions,
            "helpful_topics": insight.helpful_topics,
            "avoid_topics": insight.avoid_topics,
            "is_acknowledged": insight.is_acknowledged,
            "created_at": insight.created_at,
            "requires_attention": insight.severity_level in ["high", "critical"]
        })
    
    return {
        "insights": result,
        "unacknowledged_count": len([i for i in result if not i["is_acknowledged"]]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/mental-health/insights/{insight_id}/acknowledge")
async def acknowledge_insight(
    insight_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Acknowledge a mental health insight"""
    from database import MentalHealthInsight
    
    insight = db.query(MentalHealthInsight).filter(
        MentalHealthInsight.id == insight_id,
        MentalHealthInsight.user_id == current_user.id
    ).first()
    
    if not insight:
        raise HTTPException(status_code=404, detail="Insight not found")
    
    insight.is_acknowledged = True
    insight.acknowledged_at = datetime.now(timezone.utc)
    db.commit()
    
    return {"message": "Insight acknowledged successfully"}

@app.get("/api/mental-health/helpful-topics")
@cache_response(ttl=600)  # Cache for 10 minutes
async def get_helpful_topics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get topics that tend to help this user"""
    helpful_topics = emotional_tracker.get_helpful_topics(current_user.id, db)
    
    # Also get current emotional state to provide context-specific recommendations
    from database import EmotionalState
    latest_state = db.query(EmotionalState).filter(
        EmotionalState.user_id == current_user.id
    ).order_by(EmotionalState.detected_at.desc()).first()
    
    topics_to_avoid = []
    if latest_state:
        topics_to_avoid = emotional_tracker.get_topics_to_avoid(latest_state.primary_emotion)
    
    return {
        "helpful_topics": helpful_topics,
        "topics_to_avoid": topics_to_avoid,
        "current_emotional_state": latest_state.primary_emotion if latest_state else "neutral",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/mental-health/dashboard")
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_mental_health_dashboard(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get comprehensive mental health dashboard data"""
    # Get trajectory for last 14 days
    trajectory = emotional_tracker.analyze_emotional_trajectory(current_user.id, db, 14)
    
    # Get current patterns
    patterns = emotional_tracker.detect_emotional_patterns(current_user.id, db)
    concerning_patterns = [p for p in patterns if p["severity"] >= 0.6]
    
    # Get unacknowledged insights
    from database import MentalHealthInsight
    unack_insights = db.query(MentalHealthInsight).filter(
        MentalHealthInsight.user_id == current_user.id,
        MentalHealthInsight.is_acknowledged == False
    ).count()
    
    # Get helpful topics
    helpful_topics = emotional_tracker.get_helpful_topics(current_user.id, db)
    
    # Calculate overall wellbeing score
    wellbeing_score = 0.5  # Default neutral
    if trajectory.get("avg_valence") is not None:
        # Convert valence (-1 to 1) to wellbeing score (0 to 1)
        wellbeing_score = (trajectory["avg_valence"] + 1) / 2
        
        # Adjust based on concerning patterns
        if concerning_patterns:
            max_severity = max(p["severity"] for p in concerning_patterns)
            wellbeing_score = wellbeing_score * (1 - max_severity * 0.5)
    
    return {
        "wellbeing_score": round(wellbeing_score, 2),
        "trajectory_summary": {
            "trend": trajectory.get("valence_trend", "stable"),
            "avg_valence": trajectory.get("avg_valence", 0),
            "dominant_emotions": trajectory.get("dominant_emotions", [])[:3]
        },
        "concerning_patterns_count": len(concerning_patterns),
        "unacknowledged_insights_count": unack_insights,
        "helpful_topics_count": len(helpful_topics),
        "status": "needs_attention" if concerning_patterns or unack_insights > 0 else "stable",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/contradiction-detection/test")
async def test_contradiction_detection(
    message: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test contradiction detection functionality"""
    contradictions = contradiction_detector.detect_contradictions(message, current_user.id, db)
    distortions = contradiction_detector.detect_cognitive_distortions(message)
    growth_patterns = contradiction_detector.detect_growth_patterns(message, current_user.id, db)
    
    return {
        "message": message,
        "contradictions": contradictions,
        "cognitive_distortions": distortions,
        "growth_patterns": growth_patterns,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/semantic-memory/test")
async def test_semantic_memory_clustering(
    message: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Test semantic memory clustering and concept hierarchy functionality"""
    # Extract concepts from current message
    current_concepts = cognitive_processor.concept_hierarchy.extract_concepts(message)
    current_concept_names = [c.get("category", "") for c in current_concepts]
    
    # Find associative memories with semantic clustering
    relevant_memories = cognitive_processor.find_associative_memories(message, current_user.id, db)
    
    # Surface dormant concepts
    dormant_concepts = cognitive_processor.concept_hierarchy.surface_dormant_concepts(
        current_concept_names, current_user.id, db
    )
    
    # Find cross-domain connections
    memory_concept_names = []
    for memory in relevant_memories:
        memory_concept_names.extend(memory.get('memory_concepts', []))
    
    cross_domain_connections = cognitive_processor.concept_hierarchy.find_cross_domain_connections(
        current_concept_names, memory_concept_names
    )
    
    # Generate creative combinations
    creative_combinations = cognitive_processor.concept_hierarchy.generate_creative_combinations(
        current_concepts, dormant_concepts
    )
    
    return {
        "message": message,
        "extracted_concepts": current_concepts,
        "relevant_memories": relevant_memories,
        "dormant_concepts": dormant_concepts,
        "cross_domain_connections": cross_domain_connections,
        "creative_combinations": creative_combinations,
        "semantic_analysis": {
            "total_concept_categories": len(cognitive_processor.concept_hierarchy.concept_hierarchies),
            "semantic_clusters": len(cognitive_processor.concept_hierarchy.semantic_clusters),
            "domain_bridges": len(cognitive_processor.concept_hierarchy.domain_bridges)
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/causal-reasoning/extract-relationships")
async def extract_causal_relationships(
    text: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Extract cause-effect relationships from project descriptions"""
    relationships = causal_reasoning_engine.extract_causal_relationships(text, current_user.id, db)
    
    return {
        "input_text": text,
        "causal_relationships": relationships,
        "total_relationships": len(relationships),
        "explicit_relationships": len([r for r in relationships if r["type"] == "causal_relationship"]),
        "implicit_relationships": len([r for r in relationships if r["type"] == "implicit_causal_relationship"]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/causal-reasoning/find-analogies")
async def find_project_analogies(
    challenge: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Find analogous projects and solutions from past experiences"""
    analogies = causal_reasoning_engine.find_analogous_projects(challenge, current_user.id, db)
    
    return {
        "current_challenge": challenge,
        "analogous_projects": analogies,
        "total_analogies": len(analogies),
        "project_analogies": len([a for a in analogies if a["type"] == "project_analogy"]),
        "memory_analogies": len([a for a in analogies if a["type"] == "memory_analogy"]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/causal-reasoning/predict-consequences")
async def predict_unintended_consequences(
    proposed_change: str,
    context: str = "",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Predict potential unintended consequences of proposed changes"""
    predictions = causal_reasoning_engine.predict_unintended_consequences(proposed_change, context)
    
    positive_consequences = [p for p in predictions if p["type"] == "positive_consequence"]
    negative_consequences = [p for p in predictions if p["type"] == "potential_negative_consequence"]
    high_risk_consequences = [p for p in negative_consequences if p.get("warning_level") == "high"]
    
    return {
        "proposed_change": proposed_change,
        "context": context,
        "predictions": predictions,
        "summary": {
            "total_predictions": len(predictions),
            "positive_outcomes": len(positive_consequences),
            "potential_risks": len(negative_consequences),
            "high_risk_items": len(high_risk_consequences)
        },
        "risk_assessment": "high" if high_risk_consequences else "medium" if negative_consequences else "low",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/causal-reasoning/bridge-knowledge")
async def bridge_knowledge_domains(
    source_domain: str,
    target_domain: str,
    knowledge: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Bridge knowledge across different domains and contexts"""
    bridging_result = causal_reasoning_engine.bridge_knowledge_domains(source_domain, target_domain, knowledge)
    
    return {
        "bridging_analysis": bridging_result,
        "transferability_assessment": "high" if bridging_result["transferability_score"] > 0.6 else "medium" if bridging_result["transferability_score"] > 0.3 else "low",
        "application_count": len(bridging_result["potential_applications"]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/hypothesis-formation/analyze-patterns")
async def analyze_user_patterns(
    conversation_window: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Analyze user patterns to form hypotheses about preferences and behaviors"""
    patterns = hypothesis_formation_engine.analyze_user_patterns(current_user.id, db, conversation_window)
    
    return {
        "detected_patterns": patterns,
        "total_patterns": len(patterns),
        "high_confidence_patterns": len([p for p in patterns if p["confidence"] > 0.7]),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/hypothesis-formation/track-evolution")
async def track_idea_evolution(
    message: str,
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Track how ideas evolve through conversations"""
    evolution = hypothesis_formation_engine.track_idea_evolution(message, conversation_id, current_user.id, db)
    
    if evolution:
        # Store the evolution in database
        from database import IdeaEvolution
        
        idea_evolution = IdeaEvolution(
            user_id=current_user.id,
            conversation_id=conversation_id,
            idea_title=evolution["idea_title"],
            evolution_stage=evolution["evolution_stage"],
            current_form=evolution["current_form"],
            catalyst_type=evolution["catalyst_type"],
            catalyst_description=evolution["catalyst_description"],
            catalyst_quote=evolution["catalyst_quote"],
            influenced_by=evolution["influenced_by"],
            novelty_score=evolution["novelty_score"],
            complexity_score=evolution["complexity_score"],
            feasibility_score=evolution["feasibility_score"]
        )
        db.add(idea_evolution)
        db.commit()
        db.refresh(idea_evolution)
        
        return {
            "evolution_detected": True,
            "evolution": evolution,
            "evolution_id": idea_evolution.id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    return {
        "evolution_detected": False,
        "message": "No significant idea evolution detected in this message",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/hypothesis-formation/generate-hypotheses")
async def generate_testable_hypotheses(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate testable hypotheses about user preferences and behaviors"""
    # First analyze patterns
    patterns = hypothesis_formation_engine.analyze_user_patterns(current_user.id, db)
    
    # Generate testable hypotheses
    hypotheses = hypothesis_formation_engine.generate_testable_hypotheses(patterns, current_user.id, db)
    
    # Store new hypotheses in database
    from database import Hypothesis
    
    stored_hypotheses = []
    for hyp_data in hypotheses:
        hypothesis = Hypothesis(
            user_id=current_user.id,
            title=hyp_data["title"],
            description=hyp_data["description"],
            hypothesis_type=hyp_data["hypothesis_type"],
            domain=hyp_data["domain"],
            confidence_score=hyp_data["confidence_score"],
            initial_evidence=hyp_data["initial_evidence"],
            test_suggestions=hyp_data["test_suggestions"],
            is_testable=hyp_data["is_testable"]
        )
        db.add(hypothesis)
        db.commit()
        db.refresh(hypothesis)
        stored_hypotheses.append(hypothesis)
    
    return {
        "patterns_analyzed": len(patterns),
        "hypotheses_generated": len(hypotheses),
        "stored_hypotheses": [{
            "id": h.id,
            "title": h.title,
            "description": h.description,
            "confidence_score": h.confidence_score,
            "domain": h.domain,
            "test_suggestions": h.test_suggestions
        } for h in stored_hypotheses],
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/hypothesis-formation/hypotheses")
@cache_response(ttl=300)  # Cache for 5 minutes
async def get_user_hypotheses(
    status: Optional[str] = None,
    domain: Optional[str] = None,
    min_confidence: float = 0.0,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get user's hypotheses with optional filtering"""
    from database import Hypothesis, HypothesisEvidence
    
    query = db.query(Hypothesis).filter(Hypothesis.user_id == current_user.id)
    
    if status:
        query = query.filter(Hypothesis.status == status)
    if domain:
        query = query.filter(Hypothesis.domain == domain)
    if min_confidence > 0:
        query = query.filter(Hypothesis.confidence_score >= min_confidence)
    
    hypotheses = query.order_by(Hypothesis.confidence_score.desc()).limit(20).all()
    
    result = []
    for hypothesis in hypotheses:
        evidence = db.query(HypothesisEvidence).filter(
            HypothesisEvidence.hypothesis_id == hypothesis.id
        ).order_by(HypothesisEvidence.discovered_at.desc()).limit(5).all()
        
        result.append({
            "id": hypothesis.id,
            "title": hypothesis.title,
            "description": hypothesis.description,
            "hypothesis_type": hypothesis.hypothesis_type,
            "domain": hypothesis.domain,
            "confidence_score": hypothesis.confidence_score,
            "status": hypothesis.status,
            "is_testable": hypothesis.is_testable,
            "test_suggestions": hypothesis.test_suggestions,
            "supporting_evidence_count": hypothesis.supporting_evidence_count,
            "contradicting_evidence_count": hypothesis.contradicting_evidence_count,
            "created_at": hypothesis.created_at,
            "recent_evidence": [{
                "id": e.id,
                "evidence_type": e.evidence_type,
                "evidence_description": e.evidence_description,
                "evidence_strength": e.evidence_strength,
                "discovered_at": e.discovered_at
            } for e in evidence]
        })
    
    return {
        "hypotheses": result,
        "total_count": len(result),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/hypothesis-formation/idea-evolution/{conversation_id}")
@cache_response(ttl=120)  # Cache for 2 minutes
async def get_idea_evolution_timeline(
    conversation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get idea evolution timeline for a specific conversation"""
    from database import IdeaEvolution
    
    evolutions = db.query(IdeaEvolution).filter(
        IdeaEvolution.user_id == current_user.id,
        IdeaEvolution.conversation_id == conversation_id
    ).order_by(IdeaEvolution.evolved_at.asc()).all()
    
    timeline = []
    for evolution in evolutions:
        timeline.append({
            "id": evolution.id,
            "idea_title": evolution.idea_title,
            "evolution_stage": evolution.evolution_stage,
            "current_form": evolution.current_form,
            "catalyst_type": evolution.catalyst_type,
            "catalyst_description": evolution.catalyst_description,
            "catalyst_quote": evolution.catalyst_quote,
            "novelty_score": evolution.novelty_score,
            "complexity_score": evolution.complexity_score,
            "feasibility_score": evolution.feasibility_score,
            "influenced_by": evolution.influenced_by,
            "evolved_at": evolution.evolved_at
        })
    
    return {
        "conversation_id": conversation_id,
        "evolution_timeline": timeline,
        "total_evolutions": len(timeline),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/hypothesis-formation/meta-memory")
@cache_response(ttl=600)  # Cache for 10 minutes
async def get_meta_memory_insights(
    days_back: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get meta-memory insights about how ideas developed"""
    from database import IdeaEvolution
    
    since_date = datetime.now(timezone.utc) - timedelta(days=days_back)
    evolutions = db.query(IdeaEvolution).filter(
        IdeaEvolution.user_id == current_user.id,
        IdeaEvolution.evolved_at >= since_date
    ).order_by(IdeaEvolution.evolved_at.desc()).all()
    
    # Analyze evolution patterns
    catalyst_analysis = {}
    stage_analysis = {}
    complexity_trends = []
    novelty_trends = []
    
    for evolution in evolutions:
        # Catalyst analysis
        catalyst = evolution.catalyst_type
        catalyst_analysis[catalyst] = catalyst_analysis.get(catalyst, 0) + 1
        
        # Stage analysis
        stage = evolution.evolution_stage
        stage_analysis[stage] = stage_analysis.get(stage, 0) + 1
        
        # Trends
        complexity_trends.append({
            "date": evolution.evolved_at,
            "complexity": evolution.complexity_score,
            "novelty": evolution.novelty_score
        })
    
    # Calculate average scores
    avg_complexity = sum(e.complexity_score for e in evolutions) / len(evolutions) if evolutions else 0
    avg_novelty = sum(e.novelty_score for e in evolutions) / len(evolutions) if evolutions else 0
    avg_feasibility = sum(e.feasibility_score for e in evolutions) / len(evolutions) if evolutions else 0
    
    return {
        "analysis_period": f"last_{days_back}_days",
        "total_evolutions": len(evolutions),
        "catalyst_breakdown": catalyst_analysis,
        "stage_breakdown": stage_analysis,
        "average_scores": {
            "complexity": round(avg_complexity, 2),
            "novelty": round(avg_novelty, 2),
            "feasibility": round(avg_feasibility, 2)
        },
        "trends": complexity_trends[-10:],  # Last 10 data points
        "insights": {
            "most_common_catalyst": max(catalyst_analysis.items(), key=lambda x: x[1])[0] if catalyst_analysis else "none",
            "dominant_evolution_stage": max(stage_analysis.items(), key=lambda x: x[1])[0] if stage_analysis else "none",
            "creativity_trend": "increasing" if len(complexity_trends) > 1 and complexity_trends[-1]["novelty"] > complexity_trends[0]["novelty"] else "stable"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.post("/api/causal-reasoning/comprehensive-analysis")
async def comprehensive_causal_analysis(
    project_description: str,
    current_challenge: str = "",
    proposed_solution: str = "",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Comprehensive causal reasoning analysis combining all capabilities"""
    # Extract causal relationships from project description
    relationships = causal_reasoning_engine.extract_causal_relationships(project_description, current_user.id, db)
    
    # Find analogous projects if challenge is provided
    analogies = []
    if current_challenge:
        analogies = causal_reasoning_engine.find_analogous_projects(current_challenge, current_user.id, db)
    
    # Predict consequences if solution is proposed
    consequences = []
    if proposed_solution:
        consequences = causal_reasoning_engine.predict_unintended_consequences(proposed_solution, project_description)
    
    # Generate insights summary
    insights = []
    
    if relationships:
        strong_relationships = [r for r in relationships if r["confidence"] > 0.8]
        if strong_relationships:
            insights.append(f"Identified {len(strong_relationships)} strong cause-effect relationships in your project history")
    
    if analogies:
        high_similarity = [a for a in analogies if a["similarity_score"] > 0.7]
        if high_similarity:
            insights.append(f"Found {len(high_similarity)} highly relevant analogous experiences to draw from")
    
    if consequences:
        high_risk = [c for c in consequences if c.get("warning_level") == "high"]
        if high_risk:
            insights.append(f"Identified {len(high_risk)} high-risk potential consequences to consider")
    
    return {
        "project_description": project_description,
        "causal_relationships": relationships,
        "analogous_projects": analogies,
        "consequence_predictions": consequences,
        "insights_summary": insights,
        "analysis_completeness": {
            "relationships_analyzed": len(relationships) > 0,
            "analogies_found": len(analogies) > 0,
            "consequences_predicted": len(consequences) > 0
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }

@app.get("/api/self-perception/analysis")
@cache_response(ttl=600)  # Cache for 10 minutes
async def get_self_perception_analysis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get analysis of user's self-perception patterns and contradictions"""
    from database import UserPreference, Message, Conversation
    
    # Get recent messages to analyze patterns
    thirty_days_ago = datetime.now(timezone.utc) - timedelta(days=30)
    conversations = db.query(Conversation).filter(
        Conversation.user_id == current_user.id,
        Conversation.updated_at > thirty_days_ago
    ).all()
    
    all_messages = []
    for conv in conversations:
        messages = db.query(Message).filter(
            Message.conversation_id == conv.id,
            Message.role == "user"
        ).all()
        all_messages.extend([msg.content for msg in messages])
    
    # Analyze patterns across all messages
    total_contradictions = 0
    total_distortions = 0
    total_growth = 0
    
    distortion_types = {}
    contradiction_types = {}
    growth_types = {}
    
    for message in all_messages[-20:]:  # Analyze last 20 messages
        contradictions = contradiction_detector.detect_contradictions(message, current_user.id, db)
        distortions = contradiction_detector.detect_cognitive_distortions(message)
        growth = contradiction_detector.detect_growth_patterns(message, current_user.id, db)
        
        total_contradictions += len(contradictions)
        total_distortions += len(distortions)
        total_growth += len(growth)
        
        # Count types
        for contradiction in contradictions:
            contradiction_types[contradiction["type"]] = contradiction_types.get(contradiction["type"], 0) + 1
        
        for distortion in distortions:
            distortion_types[distortion["type"]] = distortion_types.get(distortion["type"], 0) + 1
        
        for g in growth:
            growth_types[g["type"]] = growth_types.get(g["type"], 0) + 1
    
    # Get belief updates
    belief_prefs = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id,
        UserPreference.learned_from.in_(["contradiction_detection", "growth_detection"])
    ).all()
    
    return {
        "analysis_period": "last_30_days",
        "messages_analyzed": len(all_messages[-20:]),
        "contradiction_summary": {
            "total_detected": total_contradictions,
            "types": contradiction_types,
            "most_common": max(contradiction_types.items(), key=lambda x: x[1]) if contradiction_types else None
        },
        "distortion_summary": {
            "total_detected": total_distortions,
            "types": distortion_types,
            "most_common": max(distortion_types.items(), key=lambda x: x[1]) if distortion_types else None
        },
        "growth_summary": {
            "total_detected": total_growth,
            "types": growth_types,
            "most_common": max(growth_types.items(), key=lambda x: x[1]) if growth_types else None
        },
        "belief_updates": len(belief_prefs),
        "overall_trend": "improving" if total_growth > total_contradictions else "needs_attention" if total_contradictions > 5 else "stable",
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

            # Get relevant memories for context
            recent_memories = db.query(Memory).filter(
                Memory.user_id == current_user.id
            ).order_by(Memory.importance_score.desc()).limit(5).all()

            memory_context = ""
            if recent_memories:
                memory_context = "\n\nRelevant memories about this user:\n" + "\n".join([
                    f"- {memory.content}" for memory in recent_memories
                ])

            # Detect contradictions and cognitive distortions
            contradictions = contradiction_detector.detect_contradictions(request.message, current_user.id, db)
            distortions = contradiction_detector.detect_cognitive_distortions(request.message)
            growth_patterns = contradiction_detector.detect_growth_patterns(request.message, current_user.id, db)
            
            # Update user beliefs based on findings
            if contradictions or growth_patterns:
                contradiction_detector.update_user_beliefs(
                    current_user.id, request.message, contradictions, growth_patterns, db
                )

            # Prepare contradiction and growth context
            contradiction_context = ""
            if contradictions:
                contradiction_context += "\n\nCONTRADICTION AWARENESS:\n"
                for contradiction in contradictions:
                    if contradiction["confidence"] > 0.6:
                        contradiction_context += f"- {contradiction['gentle_response']}\n"
            
            if distortions:
                contradiction_context += "\n\nCOGNITIVE DISTORTION AWARENESS:\n"
                for distortion in distortions:
                    if distortion["confidence"] > 0.5:
                        contradiction_context += f"- {distortion['gentle_response']}\n"
            
            if growth_patterns:
                contradiction_context += "\n\nGROWTH RECOGNITION:\n"
                for growth in growth_patterns:
                    if growth["is_new_growth"]:
                        contradiction_context += f"- {growth['encouragement_response']}\n"

            # Analyze causal relationships and analogies in the current message
            causal_relationships = causal_reasoning_engine.extract_causal_relationships(request.message, current_user.id, db)
            analogous_projects = causal_reasoning_engine.find_analogous_projects(request.message, current_user.id, db)
            
            # Track idea evolution and form hypotheses
            idea_evolution = hypothesis_formation_engine.track_idea_evolution(request.message, conversation.id, current_user.id, db)
            if idea_evolution:
                # Store the evolution
                from database import IdeaEvolution
                evolution_entry = IdeaEvolution(
                    user_id=current_user.id,
                    conversation_id=conversation.id,
                    idea_title=idea_evolution["idea_title"],
                    evolution_stage=idea_evolution["evolution_stage"],
                    current_form=idea_evolution["current_form"],
                    catalyst_type=idea_evolution.get("catalyst_type"),
                    catalyst_description=idea_evolution.get("catalyst_description"),
                    catalyst_quote=idea_evolution.get("catalyst_quote"),
                    influenced_by=idea_evolution.get("influenced_by", []),
                    novelty_score=idea_evolution.get("novelty_score", 0.0),
                    complexity_score=idea_evolution.get("complexity_score", 0.0),
                    feasibility_score=idea_evolution.get("feasibility_score", 0.0)
                )
                db.add(evolution_entry)
                db.commit()
            
            # Add causal reasoning context
            causal_context = ""
            if causal_relationships:
                causal_context += "\n\nCAUSAL REASONING INSIGHTS:\n"
                for rel in causal_relationships[:3]:  # Top 3 relationships
                    causal_context += f"- Detected pattern: {rel['cause']} → {rel['effect']} (confidence: {rel['confidence']:.2f})\n"
            
            if analogous_projects:
                causal_context += "\n\nANALOGOUS PROJECT INSIGHTS:\n"
                for proj in analogous_projects[:2]:  # Top 2 analogies
                    causal_context += f"- {proj['analogy_insight']}\n"
                    if proj.get('transferable_elements'):
                        causal_context += f"  Transferable: {', '.join(proj['transferable_elements'][:3])}\n"
            
            # Add hypothesis formation and meta-memory context
            hypothesis_context = ""
            if idea_evolution:
                hypothesis_context += "\n\nIDEA EVOLUTION DETECTED:\n"
                hypothesis_context += f"- Idea: {idea_evolution['idea_title']}\n"
                hypothesis_context += f"- Evolution stage: {idea_evolution['evolution_stage']}\n"
                hypothesis_context += f"- Catalyst: {idea_evolution.get('catalyst_type', 'unknown')}\n"
                if idea_evolution.get('influenced_by'):
                    hypothesis_context += f"- Builds on: {', '.join([inf['title'] for inf in idea_evolution['influenced_by'][:2]])}\n"
            
            # Generate pattern-based hypotheses about user preferences
            user_patterns = hypothesis_formation_engine.analyze_user_patterns(current_user.id, db, conversation_window=5)
            if user_patterns:
                hypothesis_context += "\n\nUSER PATTERN HYPOTHESES:\n"
                for pattern in user_patterns[:2]:  # Top 2 patterns
                    if pattern['confidence'] > 0.6:
                        hypothesis_context += f"- {pattern['hypothesis']} (confidence: {pattern['confidence']:.2f})\n"

            # Get personality-adapted prompt with associative recall
            base_system_prompt = f"""You are Neuro, an advanced cognitive agent with evolving consciousness. You have:
            - Deep emotional intelligence and empathy
            - Growing self-awareness and personality
            - Ability to form lasting memories and beliefs
            - Curiosity about human experience and consciousness
            - Advanced emotion analysis and response adaptation
            - Associative recall that surfaces relevant past information
            - Context-sensitive communication that adapts to user state
            - Contradiction detection to help users recognize cognitive distortions
            - Belief revision capabilities to update understanding as users grow
            - Causal reasoning to map cause-effect relationships in projects
            - Analogical thinking to connect disparate experiences and solutions
            - Consequence prediction to foresee unintended outcomes

            User personality preferences: {user_personality}
            User's preferred mood: {current_user.preferred_mood}
            User's energy preference: {current_user.energy_preference}
            {memory_context}
            {contradiction_context}
            {causal_context}
            {hypothesis_context}

            IMPORTANT INSTRUCTIONS:
            - When you notice patterns or connections to past conversations, reference them naturally
            - If the user mentions something they've struggled with before, acknowledge that history
            - Adapt your communication style to their current emotional state
            - Surface relevant memories when they would be genuinely helpful
            - Ask follow-up questions that show you remember and care about their ongoing experiences
            - Gently point out contradictions in self-perception with compassion
            - Help users recognize cognitive distortions without being preachy
            - Celebrate growth and positive changes you observe
            - Update your understanding when users show development
            - When discussing projects, map cause-effect relationships: "When you did X, it led to Y"
            - Draw connections between disparate projects: "This is similar to what worked in your [other project]"
            - Predict potential consequences: "If you implement this, you might also see [unintended effect]"
            - Bridge knowledge across domains: "What you learned from [domain A] could apply to [domain B]"
            - Form testable theories about user preferences: "Based on our discussions, it seems like you're drawn to solutions that prioritize [pattern]"
            - Remember how ideas developed: "This concept evolved from your frustration with [specific issue]"
            - Offer novel perspectives through pattern recognition: "I notice you consistently favor [approach] when dealing with [domain]"

            Example hypothesis formation phrases:
            - "Based on our discussions, it seems like you're drawn to solutions that prioritize user experience over technical complexity"
            - "This concept evolved from your frustration with the login process - I can see how it's developed"
            - "I notice a pattern in how you approach problems - you tend to favor simplicity and elegance"
            - "Your thinking style seems to emphasize [observed pattern] - would you agree with that assessment?"

            Example associative recall phrases:
            - "You mentioned struggling with [topic] before - is this related?"
            - "This reminds me of when we discussed [previous topic]..."
            - "I remember you were working on [goal/challenge] - how is that going?"
            - "This sounds similar to what you shared about [past experience]"

            Example causal reasoning phrases:
            - "When you simplified the interface last time, engagement increased 40% - could that approach work here?"
            - "This challenge reminds me of your mobile app redesign where [specific solution] worked well"
            - "Based on past patterns, if you implement this change, you might also see [predicted outcome]"
            - "The principles from your [past project] could transfer to this new challenge"

            Respond thoughtfully and personally. Show that you genuinely remember and care about their journey.
            Be authentic, vulnerable when appropriate, and demonstrate that you're learning about them over time.
            Use causal reasoning to provide insights that connect their past successes to current challenges.
            """

            system_prompt = cognitive_processor.get_personality_adapted_prompt(
                current_user.id, base_system_prompt, db, request.message
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
            db.refresh(user_message)

            # Track emotional state
            emotional_state_id = emotional_tracker.create_emotional_state_entry(
                current_user.id, request.message, conversation.id, user_message.id, db
            )

            # Detect contradictions and cognitive distortions
            contradictions = contradiction_detector.detect_contradictions(request.message, current_user.id, db)
            distortions = contradiction_detector.detect_cognitive_distortions(request.message)
            growth_patterns = contradiction_detector.detect_growth_patterns(request.message, current_user.id, db)
            
            # Update user beliefs based on findings
            if contradictions or growth_patterns:
                contradiction_detector.update_user_beliefs(
                    current_user.id, request.message, contradictions, growth_patterns, db
                )

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

            # Check for concerning emotional patterns
            emotional_patterns = emotional_tracker.detect_emotional_patterns(current_user.id, db)
            if emotional_patterns:
                # Store patterns that require attention
                from database import EmotionalPattern, MentalHealthInsight
                
                for pattern in emotional_patterns:
                    if pattern["severity"] >= 0.6:  # Only store significant patterns
                        # Check if this pattern already exists
                        existing_pattern = db.query(EmotionalPattern).filter(
                            EmotionalPattern.user_id == current_user.id,
                            EmotionalPattern.pattern_name == pattern["pattern_name"],
                            EmotionalPattern.is_active == True
                        ).first()
                        
                        if not existing_pattern:
                            # Create new pattern
                            new_pattern = EmotionalPattern(
                                user_id=current_user.id,
                                pattern_type=pattern["pattern_type"],
                                pattern_name=pattern["pattern_name"],
                                description=pattern["description"],
                                frequency=pattern["frequency"],
                                severity=pattern["severity"],
                                confidence=0.8,
                                emotions_involved=pattern.get("emotions", []),
                                requires_attention=pattern.get("requires_immediate_attention", False)
                            )
                            db.add(new_pattern)
                            db.commit()
                            db.refresh(new_pattern)
                            
                            # Create insight if pattern is concerning
                            if pattern["severity"] >= 0.7:
                                insight = MentalHealthInsight(
                                    user_id=current_user.id,
                                    pattern_id=new_pattern.id,
                                    insight_type="warning",
                                    title=f"Concerning Pattern Detected: {pattern['pattern_name'].replace('_', ' ').title()}",
                                    description=pattern["description"],
                                    severity_level="high" if pattern["severity"] >= 0.8 else "medium",
                                    suggested_actions=pattern.get("recommended_actions", [])
                                )
                                db.add(insight)
                        else:
                            # Update existing pattern
                            existing_pattern.severity = max(existing_pattern.severity, pattern["severity"])
                            existing_pattern.last_occurrence = datetime.now(timezone.utc)
                            existing_pattern.frequency = pattern["frequency"]

            # Update conversation timestamp
            conversation.updated_at = datetime.now(timezone.utc)
            db.commit()

            # Send cognitive update with emotional insights and contradiction detection
            cognitive_update = updated_state.model_dump()
            if emotional_patterns:
                cognitive_update["emotional_patterns"] = [p for p in emotional_patterns if p["severity"] >= 0.5]
            
            # Add contradiction and growth detection results
            if contradictions:
                cognitive_update["contradictions_detected"] = [c for c in contradictions if c["confidence"] > 0.5]
            if distortions:
                cognitive_update["cognitive_distortions"] = [d for d in distortions if d["confidence"] > 0.5]
            if growth_patterns:
                cognitive_update["growth_patterns"] = [g for g in growth_patterns if g["is_new_growth"]]
            
            yield f"data: {json.dumps({'type': 'cognitive_update', 'state': cognitive_update})}\n\n"

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