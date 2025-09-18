# Neuro v2 - Persistent Cognitive Agent

## Overview
Neuro v2 is an interactive web-based chatbot that simulates an evolving, AI-powered cognitive agent. It engages in deep, reflective conversations, remembers past interactions across browser sessions, and evolves a unique set of "beliefs" based on dialogue.

## Current State
- Successfully imported and configured for Replit environment
- Web server running on port 5000 serving static HTML application
- Deployment configured for autoscale production use
- No external dependencies required - pure vanilla HTML/CSS/JavaScript

## Project Architecture
- **Single Page Application**: Built with vanilla HTML, CSS, and JavaScript
- **Static File Serving**: Uses Python's built-in HTTP server
- **Client-side Logic**: All functionality runs in the browser
- **Local Storage**: Persistent state saved in browser's localStorage
- **Dual Mode Operation**: 
  - Demo mode with built-in responses (default)
  - Optional OpenAI API integration for dynamic responses

## Features
- Persistent memory across browser sessions
- Evolving personality and beliefs system
- Emotional analysis and mood tracking
- Real-time cognitive state dashboard
- Import/export functionality for agent state
- Modern, responsive UI with typing indicators

## Technical Details
- **Frontend**: Vanilla HTML/CSS/JavaScript (no frameworks)
- **Server**: Python HTTP server (development and production)
- **Port**: 5000 (configured for Replit proxy)
- **Host**: 0.0.0.0 (allows proxy access)
- **Deployment**: Autoscale (suitable for stateless web app)

## Recent Changes
- September 18, 2025: Initial import and Replit environment setup
- Created index.html copy of neuro.html for root access
- Configured web server workflow for development
- Set up deployment configuration for production

## User Preferences
- No specific preferences documented yet