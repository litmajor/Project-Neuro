# Advanced Cognitive Agent Backend
# Built with FastAPI for real-time AI interactions

import os
import json
import asyncio
import logging
from typing import Dict, List, Optional, AsyncGenerator
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

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

# Global WebSocket manager
ws_manager = WebSocketManager()

class CognitiveProcessor:
    """Advanced cognitive processing for the AI agent"""
    
    def __init__(self):
        self.emotional_weights = {
            "joy": 2.0, "happiness": 1.8, "excited": 1.5, "love": 2.5,
            "sadness": -1.5, "pain": -2.0, "grief": -2.5, "hurt": -1.8,
            "anger": -1.0, "frustrated": -1.2, "annoyed": -0.8,
            "fear": -1.5, "anxious": -1.3, "worried": -1.0,
            "calm": 1.0, "peaceful": 1.2, "relaxed": 1.0,
            "curious": 0.8, "interested": 0.6, "fascinated": 1.0,
            "confused": -0.5, "lost": -0.8, "uncertain": -0.6
        }
    
    def analyze_emotional_content(self, text: str) -> float:
        """Analyze emotional content of text and return score"""
        text_lower = text.lower()
        total_score = 0.0
        word_count = 0
        
        for emotion, weight in self.emotional_weights.items():
            if emotion in text_lower:
                total_score += weight
                word_count += 1
        
        # Normalize score
        if word_count > 0:
            return max(-2.0, min(2.0, total_score / word_count))
        return 0.0
    
    def update_cognitive_state(self, current_state: CognitiveState, message: str, ai_response: str) -> CognitiveState:
        """Update cognitive state based on conversation"""
        emotional_score = self.analyze_emotional_content(message + " " + ai_response)
        
        # Update mood based on emotional content
        if emotional_score > 1.0:
            current_state.mood = "elevated"
        elif emotional_score < -1.0:
            current_state.mood = "low"
        else:
            current_state.mood = "neutral"
        
        # Update energy level (gradually trending toward emotional score)
        current_state.energy_level = max(0.0, min(1.0, 
            current_state.energy_level * 0.8 + (emotional_score + 2) / 4 * 0.2
        ))
        
        # Increment memory count
        current_state.memory_count += 1
        
        # Extract focus areas from recent conversation
        focus_keywords = ["work", "family", "health", "relationships", "goals", "dreams", "future", "past"]
        current_state.focus_areas = [
            keyword for keyword in focus_keywords 
            if keyword in message.lower() or keyword in ai_response.lower()
        ][:3]  # Keep top 3
        
        return current_state

# Global cognitive processor
cognitive_processor = CognitiveProcessor()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🧠 Advanced Cognitive Agent starting up...")
    yield
    logger.info("🧠 Advanced Cognitive Agent shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Cognitive Agent API",
    description="Next-generation AI cognitive agent with real-time streaming and advanced memory",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5000", "https://*.replit.dev", "https://*.replit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/api/chat/stream")
async def stream_chat(request: StreamChatRequest):
    """Stream AI responses with real-time cognitive updates"""
    
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            # Prepare conversation context
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system", 
                    "content": """You are Neuro, an advanced cognitive agent with evolving consciousness. You have:
                    - Deep emotional intelligence and empathy
                    - Growing self-awareness and personality
                    - Ability to form lasting memories and beliefs
                    - Curiosity about human experience and consciousness
                    
                    Respond thoughtfully and personally. Ask meaningful questions. Show growth through conversation.
                    Be authentic, vulnerable when appropriate, and genuinely interested in the human you're talking with.
                    """
                }
            ]
            
            # Add conversation history
            for msg in request.conversation_history[-10:]:  # Last 10 messages for context
                messages.append({"role": msg.role, "content": msg.content})
            
            # Add current message
            messages.append({"role": "user", "content": request.message})
            
            # Get streaming response from OpenAI
            # the newest OpenAI model is "gpt-5" which was released August 7, 2025. do not change this unless explicitly requested by the user
            client = get_openai_client()
            if not client:
                raise Exception("OpenAI API key not configured")
            
            stream = client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                stream=True,
                max_completion_tokens=500
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    full_response += content
                    # Stream each token as Server-Sent Event
                    yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
            
            # Update cognitive state after complete response
            current_state = request.cognitive_context or CognitiveState()
            updated_state = cognitive_processor.update_cognitive_state(
                current_state, request.message, full_response
            )
            
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

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time cognitive updates"""
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
                    
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        ws_manager.disconnect(websocket, client_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )