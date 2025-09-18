
# Local Development Setup

## Prerequisites
- Python 3.11+
- Node.js 20+
- OpenAI API key

## Setup Steps

1. **Clone and navigate to project**
   ```bash
   git clone <your-repo>
   cd <project-name>
   ```

2. **Backend Setup**
   ```bash
   cd backend
   pip install fastapi openai uvicorn websockets pydantic python-multipart
   export OPENAI_API_KEY="your-api-key-here"
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Frontend Setup** (new terminal)
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000

## Environment Variables
- `OPENAI_API_KEY`: Required for AI functionality

## Development Notes
- The app automatically detects local vs Replit environment
- WebSocket and API calls adapt to the current environment
- Both frontend and backend support hot reload
