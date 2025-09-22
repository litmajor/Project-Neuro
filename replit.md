# Neuro v3 - Advanced Cognitive Agent

## Overview
Neuro v3 is a sophisticated full-stack application featuring a React frontend and FastAPI backend that implements an advanced AI cognitive agent. The system provides real-time AI interactions, user authentication, file uploads, collaborative conversations, and comprehensive cognitive state tracking.

## Current State
- Successfully imported and configured for Replit environment
- React frontend running on port 5000 with Vite dev server configured for Replit proxy
- FastAPI backend running on localhost:8000 with real-time WebSocket support
- PostgreSQL database provisioned and connected via DATABASE_URL
- Deployment configured for VM production use with both frontend and backend
- All Python and Node.js dependencies installed and working
- Both workflows running successfully

## Project Architecture
- **Frontend**: React 19 with TypeScript, Vite, TailwindCSS, Framer Motion
- **Backend**: FastAPI with Python 3.11, SQLAlchemy, WebSockets, OpenAI integration
- **Database**: PostgreSQL with comprehensive user, conversation, and memory models
- **Real-time Features**: WebSocket connections for live cognitive state updates
- **Authentication**: JWT-based user authentication with password hashing
- **File System**: Advanced file upload and analysis capabilities

## Features
- User registration and authentication system
- Real-time AI chat with streaming responses
- Advanced cognitive state tracking and personality adaptation
- User preference learning and personalized responses
- File upload with AI-powered analysis
- Shared conversations and collaboration features
- Message reactions and interactive UI elements
- Voice features for speech input/output
- Theme switching (light/dark mode)
- Mobile-responsive design

## Technical Details
- **Frontend**: React + TypeScript + Vite on port 5000
- **Backend**: FastAPI + SQLAlchemy on port 8000
- **Database**: PostgreSQL with environment variables configured
- **Host Configuration**: 0.0.0.0 for frontend, localhost for backend
- **Deployment**: VM deployment with both services running concurrently
- **WebSockets**: Real-time connection for cognitive state synchronization

## Recent Changes
- September 22, 2025: Complete GitHub import and Replit environment setup
- Installed all Python and Node.js dependencies (backend: FastAPI, OpenAI, SQLAlchemy; frontend: React, Vite, TailwindCSS)
- Fixed backend configuration to use localhost:8000 for Replit environment
- Fixed frontend API configuration to properly construct backend URLs for Replit domains
- Created PostgreSQL database and initialized all required tables
- Set up development workflows for both frontend (port 5000) and backend (port 8000)
- Configured production deployment settings for VM deployment target
- Both services tested and running successfully with HMR working

## User Preferences
- No specific preferences documented yet

## Notes
- OpenAI API key not configured - AI functionality will use fallback responses
- WebSocket connections tested and working on backend
- Frontend successfully displays login interface and is ready for use
- All core functionality is operational and ready for development