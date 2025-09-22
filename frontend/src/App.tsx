import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Brain,
  Send,
  Settings,
  Activity,
  Heart,
  Zap,
  MessageCircle,
  Sparkles,
  LogOut,
  User as UserIcon,
  Users,
  Menu
} from 'lucide-react';
import './App.css';
import { useAuth } from './auth';
import LoginForm from './LoginForm';
import LandingPage from './LandingPage';
import { FileUpload } from './FileUpload';
import { SharedConversations } from './SharedConversations';
import { MessageReactions } from './MessageReactions';
import { ThemeProvider, ThemeToggle, useTheme } from './ThemeProvider';
import { VoiceFeatures } from './VoiceFeatures';

// Types for our advanced cognitive agent
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

interface CognitiveState {
  mood: 'elevated' | 'neutral' | 'low';
  energy_level: number;
  focus_areas: string[];
  personality_traits: Record<string, number>;
  memory_count: number;
  beliefs: string[];
}

interface WebSocketMessage {
  type: 'cognitive_update' | 'pong' | 'cognitive_state';
  state?: CognitiveState;
  timestamp?: string;
}

const AppContent: React.FC = () => {
  const { user, isAuthenticated, login, register, logout, updatePersonality } = useAuth();
  const { theme } = useTheme();
  const [showLogin, setShowLogin] = useState(false);

  // State management
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [cognitiveState, setCognitiveState] = useState<CognitiveState>({
    mood: 'neutral',
    energy_level: 0.5,
    focus_areas: [],
    personality_traits: {},
    memory_count: 0,
    beliefs: []
  });
  const [showSettings, setShowSettings] = useState(false);
  const [conversations, setConversations] = useState<any[]>([]);
  const [currentSharedConversation, setCurrentSharedConversation] = useState<number | null>(null);
  const [showFileUpload, setShowFileUpload] = useState(false);
  const [showCollaborations, setShowCollaborations] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [messageReactions, setMessageReactions] = useState<Record<string, Record<string, number>>>({});
  const [mentalHealthData, setMentalHealthData] = useState<any>(null);
  const [showMentalHealthPanel, setShowMentalHealthPanel] = useState(false);

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const currentStreamingMessage = useRef<string>('');

  // WebSocket connection for real-time updates
  useEffect(() => {
    if (!isAuthenticated) return;

    const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const isLocal = window.location.hostname === 'localhost';
    const host = window.location.host;
    const backendHost = isLocal ? 'localhost:8000' : '8000-ef06-4315-812e-4f8a24e40d06-00-4d9n4n4rocto.worf.replit.dev';
    const wsProtocol = isLocal ? 'ws' : 'wss';
    const wsUrl = `${wsProtocol}://${backendHost}/ws/${clientId}`;

    console.log('🔗 WebSocket URL:', wsUrl);

    const connectWebSocket = () => {
      try {
        wsRef.current = new WebSocket(wsUrl);

        wsRef.current.onopen = () => {
          console.log('🔗 WebSocket connected');
          setIsConnected(true);

          // Send initial ping
          if (wsRef.current) {
            wsRef.current.send(JSON.stringify({ type: 'ping' }));
          }
        };

        wsRef.current.onmessage = (event) => {
          try {
            const data: WebSocketMessage = JSON.parse(event.data);

            if (data.type === 'cognitive_update' || data.type === 'cognitive_state') {
              if (data.state) {
                setCognitiveState(data.state);
              }
            }
          } catch (error) {
            console.error('Error parsing WebSocket message:', error);
          }
        };

        wsRef.current.onclose = () => {
          console.log('🔌 WebSocket disconnected');
          setIsConnected(false);

          // Reconnect after 3 seconds
          setTimeout(connectWebSocket, 3000);
        };

        wsRef.current.onerror = (error) => {
          console.error('❌ WebSocket error:', error);
          setIsConnected(false);
        };
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        setTimeout(connectWebSocket, 3000);
      }
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [isAuthenticated]);

  // Load initial greeting
  useEffect(() => {
    if (isAuthenticated && messages.length === 0) {
      const welcomeMessage: Message = {
        id: 'welcome_msg',
        role: 'assistant',
        content: `Hello ${user?.username}! I'm Neuro, your advanced cognitive assistant. I have persistent memory and evolve through our conversations. How can I help you today?`,
        timestamp: new Date()
      };
      setMessages([welcomeMessage]);
    }
  }, [isAuthenticated, user, messages.length]);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message to AI with streaming response
  const sendMessage = async () => {
    if (!inputValue.trim() || isStreaming || !isAuthenticated) return;

    const userMessage: Message = {
      id: `msg_${Date.now()}`,
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsStreaming(true);

    // Create streaming assistant message
    const assistantMessageId = `msg_${Date.now() + 1}`;
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      timestamp: new Date(),
      isStreaming: true
    };

    setMessages(prev => [...prev, assistantMessage]);
    currentStreamingMessage.current = '';

    try {
      const isLocal = window.location.hostname === 'localhost';
      const host = window.location.host;
      const backendHost = isLocal ? 'localhost:8000' : '8000-ef06-4315-812e-4f8a24e40d06-00-4d9n4n4rocto.worf.replit.dev';
      const protocol = isLocal ? 'http' : 'https';
      const response = await fetch(`${protocol}://${backendHost}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({
          message: userMessage.content,
          conversation_history: messages.slice(-10), // Last 10 messages for context
          cognitive_context: cognitiveState
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || 'Failed to get AI response');
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response stream available');
      }

      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'token') {
                currentStreamingMessage.current += data.content;

                // Update the streaming message
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMessageId
                    ? { ...msg, content: currentStreamingMessage.current }
                    : msg
                ));
              } else if (data.type === 'cognitive_update') {
                setCognitiveState(data.state);
              } else if (data.type === 'complete') {
                // Mark streaming as complete
                setMessages(prev => prev.map(msg =>
                  msg.id === assistantMessageId
                    ? { ...msg, isStreaming: false }
                    : msg
                ));
              } else if (data.type === 'error') {
                console.error('Stream error:', data.message);
                break;
              }
            } catch (error) {
              console.error('Error parsing stream data:', error);
            }
          }
        }
      }
    } catch (error: any) {
      console.error('Error sending message:', error);

      // Update message with error
      setMessages(prev => prev.map(msg =>
        msg.id === assistantMessageId
          ? {
              ...msg,
              content: `❌ ${error.message}`,
              isStreaming: false
            }
          : msg
      ));
    } finally {
      setIsStreaming(false);
    }
  };

  // Handle Enter key press
  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Handle voice input
  const handleVoiceInput = (text: string) => {
    setInputValue(text);
  };

  // Handle message reactions
  const handleReaction = (messageId: string, reaction: string) => {
    setMessageReactions(prev => ({
      ...prev,
      [messageId]: {
        ...prev[messageId],
        [reaction]: (prev[messageId]?.[reaction] || 0) + 1
      }
    }));
  };

  // Fetch mental health dashboard data
  const fetchMentalHealthData = async () => {
    try {
      const isLocal = window.location.hostname === 'localhost';
      const host = window.location.host;
      const backendHost = isLocal ? 'localhost:8000' : '8000-ef06-4315-812e-4f8a24e40d06-00-4d9n4n4rocto.worf.replit.dev';
      const protocol = isLocal ? 'http' : 'https';
      
      const response = await fetch(`${protocol}://${backendHost}/api/mental-health/dashboard`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setMentalHealthData(data);
      }
    } catch (error) {
      console.error('Error fetching mental health data:', error);
    }
  };

  // Fetch mental health data on auth
  useEffect(() => {
    if (isAuthenticated) {
      fetchMentalHealthData();
      // Refresh every 5 minutes
      const interval = setInterval(fetchMentalHealthData, 5 * 60 * 1000);
      return () => clearInterval(interval);
    }
  }, [isAuthenticated]);

  // Test associative recall
  const testAssociativeRecall = async (testMessage: string) => {
    try {
      const isLocal = window.location.hostname === 'localhost';
      const host = window.location.host;
      const backendHost = isLocal ? 'localhost:8000' : '8000-ef06-4315-812e-4f8a24e40d06-00-4d9n4n4rocto.worf.replit.dev';
      const protocol = isLocal ? 'http' : 'https';
      
      const response = await fetch(`${protocol}://${backendHost}/api/associative-recall/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({ message: testMessage })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('🧠 Associative Recall Test Result:', result);
        
        // Add a test message showing the recall capabilities
        const testResultMessage: Message = {
          id: `test_${Date.now()}`,
          role: 'assistant',
          content: `🧠 **Associative Recall Test**\n\n**User State Detected:** ${result.user_state.emotional_state} (intensity: ${result.user_state.emotional_intensity})\n\n**Relevant Memories Found:** ${result.relevant_memories.length}\n${result.relevant_memories.map((mem: any) => `- ${mem.content.substring(0, 100)}...`).join('\n')}\n\n**Related Conversations:** ${result.related_conversations.length}\n${result.related_conversations.map((conv: any) => `- ${conv.title}`).join('\n')}\n\n**Contextual Guidance Generated:** ${result.contextual_guidance ? 'Yes' : 'No'}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, testResultMessage]);
      }
    } catch (error) {
      console.error('Error testing associative recall:', error);
    }
  };

  // Test contradiction detection
  const testContradictionDetection = async (testMessage: string) => {
    try {
      const isLocal = window.location.hostname === 'localhost';
      const host = window.location.host;
      const backendHost = isLocal ? 'localhost:8000' : '8000-ef06-4315-812e-4f8a24e40d06-00-4d9n4n4rocto.worf.replit.dev';
      const protocol = isLocal ? 'http' : 'https';
      
      const response = await fetch(`${protocol}://${backendHost}/api/contradiction-detection/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({ message: testMessage })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('🔍 Contradiction Detection Test Result:', result);
        
        // Add a test message showing the contradiction detection capabilities
        const testResultMessage: Message = {
          id: `contradiction_test_${Date.now()}`,
          role: 'assistant',
          content: `🔍 **Contradiction Detection Test**\n\n**Contradictions Found:** ${result.contradictions.length}\n${result.contradictions.map((cont: any) => `- ${cont.type}: ${cont.gentle_response}`).join('\n')}\n\n**Cognitive Distortions:** ${result.cognitive_distortions.length}\n${result.cognitive_distortions.map((dist: any) => `- ${dist.type}: ${dist.gentle_response}`).join('\n')}\n\n**Growth Patterns:** ${result.growth_patterns.length}\n${result.growth_patterns.map((growth: any) => `- ${growth.type}: ${growth.encouragement_response}`).join('\n')}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, testResultMessage]);
      }
    } catch (error) {
      console.error('Error testing contradiction detection:', error);
    }
  };

  // Test semantic memory clustering
  const testSemanticMemory = async (testMessage: string) => {
    try {
      const isLocal = window.location.hostname === 'localhost';
      const host = window.location.host;
      const backendHost = isLocal ? 'localhost:8000' : '8000-ef06-4315-812e-4f8a24e40d06-00-4d9n4n4rocto.worf.replit.dev';
      const protocol = isLocal ? 'http' : 'https';
      
      const response = await fetch(`${protocol}://${backendHost}/api/semantic-memory/test`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({ message: testMessage })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('🧠 Semantic Memory Test Result:', result);
        
        // Add a test message showing the semantic memory capabilities
        const testResultMessage: Message = {
          id: `semantic_test_${Date.now()}`,
          role: 'assistant',
          content: `🧠 **Semantic Memory Clustering Test**\n\n**Extracted Concepts:** ${result.extracted_concepts.length}\n${result.extracted_concepts.map((concept: any) => `- ${concept.category} (strength: ${concept.activation_strength?.toFixed(2)})`).join('\n')}\n\n**Semantic Memories Found:** ${result.relevant_memories.length}\n${result.relevant_memories.map((mem: any) => `- ${mem.category}: ${mem.content.substring(0, 80)}... (semantic: ${mem.semantic_score?.toFixed(2)})`).join('\n')}\n\n**Dormant Concepts Surfaced:** ${result.dormant_concepts.length}\n${result.dormant_concepts.map((dormant: any) => `- ${dormant.concept_category}: ${dormant.reactivation_reason} (${dormant.days_dormant} days old)`).join('\n')}\n\n**Cross-Domain Connections:** ${result.cross_domain_connections.length}\n${result.cross_domain_connections.map((conn: any) => `- ${conn.insight}`).join('\n')}\n\n**Creative Combinations:** ${result.creative_combinations.length}\n${result.creative_combinations.map((combo: any) => `- ${combo.creative_insight}`).join('\n')}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, testResultMessage]);
      }
    } catch (error) {
      console.error('Error testing semantic memory:', error);
    }
  };

  // Test causal reasoning
  const testCausalReasoning = async (testMessage: string) => {
    try {
      const isLocal = window.location.hostname === 'localhost';
      const host = window.location.host;
      const backendHost = isLocal ? 'localhost:8000' : '8000-ef06-4315-812e-4f8a24e40d06-00-4d9n4n4rocto.worf.replit.dev';
      const protocol = isLocal ? 'http' : 'https';
      
      const response = await fetch(`${protocol}://${backendHost}/api/causal-reasoning/comprehensive-analysis`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({ 
          project_description: testMessage,
          current_challenge: "How to improve user engagement",
          proposed_solution: "Simplify the interface and add gamification elements"
        })
      });
      
      if (response.ok) {
        const result = await response.json();
        console.log('🔗 Causal Reasoning Test Result:', result);
        
        // Add a test message showing the causal reasoning capabilities
        const testResultMessage: Message = {
          id: `causal_test_${Date.now()}`,
          role: 'assistant',
          content: `🔗 **Causal Reasoning + Analogical Thinking Test**\n\n**Cause-Effect Relationships:** ${result.causal_relationships.length}\n${result.causal_relationships.map((rel: any) => `- ${rel.cause} → ${rel.effect} (confidence: ${Math.round(rel.confidence * 100)}%)`).join('\n')}\n\n**Analogous Projects:** ${result.analogous_projects.length}\n${result.analogous_projects.map((proj: any) => `- ${proj.source_domain} → ${proj.target_domain}: ${proj.analogy_insight.substring(0, 100)}...`).join('\n')}\n\n**Predicted Consequences:** ${result.consequence_predictions.length}\n${result.consequence_predictions.map((cons: any) => `- ${cons.type === 'positive_consequence' ? '✅' : '⚠️'} ${cons.consequence} (${Math.round(cons.probability * 100)}% likely)`).join('\n')}\n\n**Key Insights:**\n${result.insights_summary.map((insight: string) => `- ${insight}`).join('\n')}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, testResultMessage]);
      }
    } catch (error) {
      console.error('Error testing causal reasoning:', error);
    }
  };

  // Get mood color based on cognitive state
  const getMoodColor = (mood: string) => {
    switch (mood) {
      case 'elevated': return 'text-green-400';
      case 'low': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  // Get mood emoji
  const getMoodEmoji = (mood: string) => {
    switch (mood) {
      case 'elevated': return '😊';
      case 'low': return '😔';
      default: return '😐';
    }
  };

  // Handle joining shared conversation
  const handleJoinSharedConversation = (conversationId: number) => {
    setCurrentSharedConversation(conversationId);
    setShowCollaborations(false);
    // Clear current messages to load shared conversation
    setMessages([]);
  };

  // Handle file upload
  const handleFileUploaded = (file: any) => {
    // Add file reference to messages
    const fileMessage: Message = {
      id: `file_${Date.now()}`,
      role: 'user',
      content: `📎 Uploaded: ${file.filename}\n\n${file.analysis?.description || file.analysis?.analysis || 'File uploaded successfully'}`,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, fileMessage]);
  };

  // Show landing page or login form if not authenticated
  if (!isAuthenticated) {
    if (showLogin) {
      return <LoginForm onLogin={login} onRegister={register} />;
    }
    return <LandingPage onGetStarted={() => setShowLogin(true)} />;
  }

  return (
    <div className={`min-h-screen transition-colors font-sans ${
      theme === 'light'
        ? 'bg-gradient-to-br from-gray-50 via-blue-50 to-purple-50 text-gray-900'
        : 'bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white'
    }`}>
      {/* Header */}
      <motion.header
        className={`backdrop-blur-md border-b p-4 ${
          theme === 'light'
            ? 'bg-white/80 border-gray-200'
            : 'bg-white/10 border-white/20'
        }`}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="max-w-7xl mx-auto flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <motion.div
              className="p-2 bg-blue-500/20 rounded-lg"
              animate={{
                scale: [1, 1.05, 1],
                rotate: [0, 5, -5, 0]
              }}
              transition={{
                duration: 4,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            >
              <Brain className="w-8 h-8 text-blue-400" />
            </motion.div>
            <div className="hidden sm:block">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Neuro Assistant
              </h1>
              <p className={`text-sm ${theme === 'light' ? 'text-gray-600' : 'text-gray-400'}`}>
                Your Persistent AI Companion
              </p>
            </div>
          </div>

          <div className="flex items-center space-x-2 md:space-x-4">
            {/* Mobile menu button */}
            <motion.button
              className={`md:hidden p-2 rounded-lg transition-colors ${
                theme === 'light' ? 'hover:bg-gray-200' : 'hover:bg-white/10'
              }`}
              onClick={() => setSidebarOpen(!sidebarOpen)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Menu className="w-5 h-5" />
            </motion.button>
            {/* User Info */}
            <div className={`hidden sm:flex items-center space-x-2 text-sm ${
              theme === 'light' ? 'text-gray-700' : 'text-gray-300'
            }`}>
              <UserIcon className="w-4 h-4 text-blue-400" />
              <span>Welcome, {user?.username}</span>
            </div>

            {/* Theme Toggle */}
            <ThemeToggle />

            {/* Mental Health Status */}
            {mentalHealthData && (
              <motion.button
                className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  mentalHealthData.status === 'needs_attention'
                    ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30'
                    : 'bg-green-500/20 text-green-400 hover:bg-green-500/30'
                }`}
                onClick={() => setShowMentalHealthPanel(!showMentalHealthPanel)}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Heart className={`w-3 h-3 ${
                  mentalHealthData.status === 'needs_attention' ? 'text-yellow-400' : 'text-green-400'
                }`} />
                <span>{Math.round(mentalHealthData.wellbeing_score * 100)}%</span>
                {mentalHealthData.unacknowledged_insights_count > 0 && (
                  <span className="bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center">
                    {mentalHealthData.unacknowledged_insights_count}
                  </span>
                )}
              </motion.button>
            )}

            {/* Connection Status */}
            <motion.div
              className={`flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-medium ${
                isConnected
                  ? 'bg-green-500/20 text-green-400'
                  : 'bg-red-500/20 text-red-400'
              }`}
              animate={{ opacity: [0.7, 1, 0.7] }}
              transition={{ duration: 2, repeat: Infinity }}
            >
              <div className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-400' : 'bg-red-400'
              }`} />
              {isConnected ? 'CONNECTED' : 'CONNECTING'}
            </motion.div>

            {/* Logout Button */}
            <motion.button
              className="p-2 hover:bg-red-500/20 text-red-400 rounded-lg transition-colors"
              onClick={logout}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <LogOut className="w-5 h-5" />
            </motion.button>
          </div>
        </div>
      </motion.header>

      <div className="max-w-7xl mx-auto flex h-[calc(100vh-80px)] relative">
        {/* Mobile sidebar overlay */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 z-40 md:hidden"
              onClick={() => setSidebarOpen(false)}
            />
          )}
        </AnimatePresence>

        {/* Main Chat Area */}
        <div className="flex-1 flex flex-col min-w-0">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            <AnimatePresence>
              {messages.map((message) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                  className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div className={`max-w-2xl ${
                    message.role === 'user'
                      ? 'bg-blue-500/20 border border-blue-500/30'
                      : 'bg-white/10 border border-white/20'
                  } rounded-2xl p-4 backdrop-blur-md`}>
                    <div className="flex items-center space-x-2 mb-2">
                      <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                        message.role === 'user'
                          ? 'bg-blue-500 text-white'
                          : 'bg-purple-500 text-white'
                      }`}>
                        {message.role === 'user' ? 'U' : 'N'}
                      </div>
                      <span className="text-xs text-gray-400">
                        {message.timestamp.toLocaleTimeString()}
                      </span>
                      {message.isStreaming && (
                        <motion.div
                          className="flex items-center space-x-2"
                        >
                          <motion.div
                            className="flex space-x-1"
                            animate={{ opacity: [0.5, 1, 0.5] }}
                            transition={{ duration: 1, repeat: Infinity }}
                          >
                            <div className="w-1 h-1 bg-blue-400 rounded-full" />
                            <div className="w-1 h-1 bg-blue-400 rounded-full" />
                            <div className="w-1 h-1 bg-blue-400 rounded-full" />
                          </motion.div>
                          <span className="text-xs text-purple-400">
                            🧠 Recalling context
                          </span>
                        </motion.div>
                      )}
                    </div>
                    <div className={`prose max-w-none ${theme === 'light' ? 'prose-gray' : 'prose-invert'}`}>
                      {message.content}
                    </div>

                    {/* Message reactions */}
                    {message.role === 'assistant' && !message.isStreaming && (
                      <div className="mt-3 pt-2 border-t border-white/10">
                        <MessageReactions
                          messageId={message.id}
                          onReaction={handleReaction}
                          existingReactions={messageReactions[message.id]}
                        />
                      </div>
                    )}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <motion.div
            className={`p-4 md:p-6 backdrop-blur-md border-t ${
              theme === 'light'
                ? 'bg-gray-50/80 border-gray-200'
                : 'bg-white/5 border-white/20'
            }`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <div className="flex items-center space-x-2 md:space-x-4">
              <VoiceFeatures
                onVoiceInput={handleVoiceInput}
                isListening={isListening}
                setIsListening={setIsListening}
              />

              <div className="flex-1 relative">
                <textarea
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Share your thoughts with Neuro..."
                  className={`w-full p-4 border rounded-2xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors ${
                    theme === 'light'
                      ? 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                      : 'bg-white/10 border-white/20 text-white placeholder-gray-400'
                  }`}
                  rows={window.innerWidth < 768 ? 1 : 2}
                  disabled={isStreaming}
                />
              </div>

              <motion.button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isStreaming}
                className="p-3 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed rounded-full transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isStreaming ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Sparkles className="w-5 h-5" />
                  </motion.div>
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </motion.button>
            </div>
          </motion.div>
        </div>

        {/* Cognitive State Sidebar */}
        <motion.div
          className={`${
            sidebarOpen ? 'translate-x-0' : 'translate-x-full'
          } fixed md:relative md:translate-x-0 top-0 right-0 w-80 h-full backdrop-blur-md border-l p-4 md:p-6 overflow-y-auto z-50 transition-transform ${
            theme === 'light'
              ? 'bg-gray-50/90 border-gray-200'
              : 'bg-gray-900/90 border-white/20'
          }`}
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
        >
          <h2 className="text-xl font-bold mb-6 flex items-center space-x-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <span>Cognitive State</span>
          </h2>

          {/* Semantic Memory System */}
          <motion.div
            className="bg-white/10 rounded-xl p-4 mb-6"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="font-semibold mb-2 flex items-center space-x-2">
              <Brain className="w-4 h-4 text-purple-400" />
              <span>Semantic Memory</span>
            </h3>
            <div className="space-y-3">
              {/* Concept Hierarchies */}
              <div className="text-center">
                <div className="text-xs text-gray-400 mb-1">Concept Hierarchies</div>
                <motion.div
                  className="w-full bg-white/20 rounded-full h-2"
                  initial={{ width: 0 }}
                  animate={{ width: "100%" }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  <motion.div
                    className="bg-gradient-to-r from-purple-400 to-blue-400 h-2 rounded-full"
                    animate={{ 
                      x: [-100, 100, -100],
                      opacity: [0.3, 1, 0.3]
                    }}
                    transition={{
                      duration: 3,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                  />
                </motion.div>
                <div className="text-xs text-purple-400 mt-1">
                  Mapping abstract relationships
                </div>
              </div>
              
              {/* Cross-Domain Bridges */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">Cross-Domain Bridges</span>
                <motion.span 
                  className="text-blue-400"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  🌉 Active
                </motion.span>
              </div>
              
              {/* Dormant Concept Detection */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">Dormant Concepts</span>
                <motion.span 
                  className="text-green-400"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2.5, repeat: Infinity }}
                >
                  💡 Surfacing
                </motion.span>
              </div>
              
              {/* Test Button */}
              <motion.button
                className="w-full text-xs bg-purple-500/20 hover:bg-purple-500/30 text-purple-400 px-2 py-1 rounded transition-colors mb-2"
                onClick={() => testSemanticMemory("I'm working on a creative project that combines nature patterns with technology design")}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Test Semantic Memory
              </motion.button>
            </div>
          </motion.div>

          {/* Causal Reasoning System */}
          <motion.div
            className="bg-white/10 rounded-xl p-4 mb-6"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="font-semibold mb-2 flex items-center space-x-2">
              <Activity className="w-4 h-4 text-orange-400" />
              <span>Causal Reasoning</span>
            </h3>
            <div className="space-y-3">
              {/* Cause-Effect Mapping */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">Cause-Effect Mapping</span>
                <motion.span 
                  className="text-orange-400"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2, repeat: Infinity }}
                >
                  🔗 Analyzing
                </motion.span>
              </div>
              
              {/* Analogical Thinking */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">Analogical Thinking</span>
                <motion.span 
                  className="text-cyan-400"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2.3, repeat: Infinity }}
                >
                  🔄 Connecting
                </motion.span>
              </div>
              
              {/* Consequence Prediction */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-gray-400">Consequence Prediction</span>
                <motion.span 
                  className="text-yellow-400"
                  animate={{ opacity: [0.5, 1, 0.5] }}
                  transition={{ duration: 2.7, repeat: Infinity }}
                >
                  ⚠️ Foreseeing
                </motion.span>
              </div>
              
              {/* Test Button */}
              <motion.button
                className="w-full text-xs bg-orange-500/20 hover:bg-orange-500/30 text-orange-400 px-2 py-1 rounded transition-colors"
                onClick={() => testCausalReasoning("When we simplified the interface last time, user engagement increased 40%. This mobile app redesign challenge is similar to our web optimization project.")}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Test Causal Reasoning
              </motion.button>
            </div>
          </motion.div>

          {/* Mood Display */}
          <motion.div
            className="bg-white/10 rounded-xl p-4 mb-6"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="font-semibold mb-2 flex items-center space-x-2">
              <Heart className="w-4 h-4 text-pink-400" />
              <span>Current Mood</span>
            </h3>
            <div className="text-center">
              <div className="text-3xl mb-2">{getMoodEmoji(cognitiveState.mood)}</div>
              <div className={`font-bold text-lg capitalize ${getMoodColor(cognitiveState.mood)}`}>
                {cognitiveState.mood}
              </div>
            </div>
          </motion.div>

          {/* Energy Level */}
          <motion.div
            className="bg-white/10 rounded-xl p-4 mb-6"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="font-semibold mb-3 flex items-center space-x-2">
              <Zap className="w-4 h-4 text-yellow-400" />
              <span>Energy Level</span>
            </h3>
            <div className="relative">
              <div className="w-full bg-white/20 rounded-full h-3">
                <motion.div
                  className="bg-gradient-to-r from-yellow-400 to-orange-400 h-3 rounded-full"
                  initial={{ width: 0 }}
                  animate={{ width: `${cognitiveState.energy_level * 100}%` }}
                  transition={{ duration: 1, ease: "easeOut" }}
                />
              </div>
              <div className="text-center mt-2 text-sm text-gray-400">
                {Math.round(cognitiveState.energy_level * 100)}%
              </div>
            </div>
          </motion.div>

          {/* Focus Areas */}
          <motion.div
            className="bg-white/10 rounded-xl p-4 mb-6"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="font-semibold mb-3 flex items-center space-x-2">
              <Brain className="w-4 h-4 text-blue-400" />
              <span>Focus Areas</span>
            </h3>
            <div className="space-y-2">
              {cognitiveState.focus_areas.length > 0 ? (
                cognitiveState.focus_areas.map((area, index) => (
                  <motion.div
                    key={area}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="bg-blue-500/20 text-blue-400 px-3 py-1 rounded-full text-sm capitalize"
                  >
                    {area}
                  </motion.div>
                ))
              ) : (
                <div className="text-gray-400 text-sm">No active focus areas</div>
              )}
            </div>
          </motion.div>

          {/* Memory Count */}
          <motion.div
            className="bg-white/10 rounded-xl p-4 mb-6"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="font-semibold mb-2 flex items-center space-x-2">
              <MessageCircle className="w-4 h-4 text-green-400" />
              <span>Memories</span>
            </h3>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-400">
                {cognitiveState.memory_count}
              </div>
              <div className="text-sm text-gray-400">Stored interactions</div>
            </div>
          </motion.div>

          {/* Contradiction Detection Panel */}
          {cognitiveState.contradictions_detected && cognitiveState.contradictions_detected.length > 0 && (
            <motion.div
              className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-4 mb-6"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              whileHover={{ scale: 1.02 }}
            >
              <h3 className="font-semibold mb-3 flex items-center space-x-2">
                <Brain className="w-4 h-4 text-yellow-400" />
                <span>Self-Perception Insights</span>
              </h3>
              <div className="space-y-2">
                {cognitiveState.contradictions_detected.map((contradiction, index) => (
                  <div key={index} className="bg-yellow-400/20 rounded-lg p-3">
                    <div className="text-sm font-medium text-yellow-400 mb-1">
                      Pattern Noticed
                    </div>
                    <div className="text-xs text-yellow-300">
                      {contradiction.gentle_response}
                    </div>
                    <div className="text-xs text-yellow-200 mt-1 opacity-70">
                      Confidence: {Math.round(contradiction.confidence * 100)}%
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Cognitive Distortion Detection */}
          {cognitiveState.cognitive_distortions && cognitiveState.cognitive_distortions.length > 0 && (
            <motion.div
              className="bg-orange-500/10 border border-orange-500/30 rounded-xl p-4 mb-6"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              whileHover={{ scale: 1.02 }}
            >
              <h3 className="font-semibold mb-3 flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-orange-400" />
                <span>Thinking Patterns</span>
              </h3>
              <div className="space-y-2">
                {cognitiveState.cognitive_distortions.map((distortion, index) => (
                  <div key={index} className="bg-orange-400/20 rounded-lg p-3">
                    <div className="text-sm font-medium text-orange-400 mb-1">
                      {distortion.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div className="text-xs text-orange-300">
                      {distortion.gentle_response}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Growth Recognition */}
          {cognitiveState.growth_patterns && cognitiveState.growth_patterns.length > 0 && (
            <motion.div
              className="bg-green-500/10 border border-green-500/30 rounded-xl p-4 mb-6"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              whileHover={{ scale: 1.02 }}
            >
              <h3 className="font-semibold mb-3 flex items-center space-x-2">
                <Sparkles className="w-4 h-4 text-green-400" />
                <span>Growth Detected! 🌱</span>
              </h3>
              <div className="space-y-2">
                {cognitiveState.growth_patterns.map((growth, index) => (
                  <div key={index} className="bg-green-400/20 rounded-lg p-3">
                    <div className="text-sm font-medium text-green-400 mb-1">
                      {growth.type.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                    </div>
                    <div className="text-xs text-green-300">
                      {growth.encouragement_response}
                    </div>
                  </div>
                ))}
              </div>
            </motion.div>
          )}

          {/* Mental Health Panel */}
          {showMentalHealthPanel && mentalHealthData && (
            <motion.div
              className="bg-white/10 rounded-xl p-4 mb-6"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              whileHover={{ scale: 1.02 }}
            >
              <h3 className="font-semibold mb-3 flex items-center space-x-2">
                <Heart className="w-4 h-4 text-pink-400" />
                <span>Mental Health Insights</span>
              </h3>
              
              {/* Wellbeing Score */}
              <div className="mb-4">
                <div className="flex justify-between text-sm mb-1">
                  <span>Wellbeing Score</span>
                  <span className={`font-bold ${
                    mentalHealthData.wellbeing_score >= 0.7 ? 'text-green-400' :
                    mentalHealthData.wellbeing_score >= 0.4 ? 'text-yellow-400' : 'text-red-400'
                  }`}>
                    {Math.round(mentalHealthData.wellbeing_score * 100)}%
                  </span>
                </div>
                <div className="w-full bg-white/20 rounded-full h-2">
                  <motion.div
                    className={`h-2 rounded-full ${
                      mentalHealthData.wellbeing_score >= 0.7 ? 'bg-green-400' :
                      mentalHealthData.wellbeing_score >= 0.4 ? 'bg-yellow-400' : 'bg-red-400'
                    }`}
                    initial={{ width: 0 }}
                    animate={{ width: `${mentalHealthData.wellbeing_score * 100}%` }}
                    transition={{ duration: 1, ease: "easeOut" }}
                  />
                </div>
              </div>

              {/* Emotional Trend */}
              <div className="mb-4">
                <div className="text-sm text-gray-400 mb-1">Emotional Trend (14 days)</div>
                <div className={`text-sm font-medium ${
                  mentalHealthData.trajectory_summary.trend === 'improving' ? 'text-green-400' :
                  mentalHealthData.trajectory_summary.trend === 'declining' ? 'text-red-400' : 'text-gray-400'
                }`}>
                  {mentalHealthData.trajectory_summary.trend === 'improving' ? '📈 Improving' :
                   mentalHealthData.trajectory_summary.trend === 'declining' ? '📉 Declining' : '➡️ Stable'}
                </div>
              </div>

              {/* Dominant Emotions */}
              {mentalHealthData.trajectory_summary.dominant_emotions.length > 0 && (
                <div className="mb-4">
                  <div className="text-sm text-gray-400 mb-2">Recent Emotions</div>
                  <div className="flex flex-wrap gap-2">
                    {mentalHealthData.trajectory_summary.dominant_emotions.map(([emotion, count], index) => (
                      <span
                        key={emotion}
                        className="bg-blue-500/20 text-blue-400 px-2 py-1 rounded text-xs"
                      >
                        {emotion} ({count})
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Alerts */}
              {mentalHealthData.concerning_patterns_count > 0 && (
                <div className="bg-yellow-500/20 border border-yellow-500/30 rounded-lg p-3 mb-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-yellow-400">⚠️</span>
                    <span className="text-sm font-medium text-yellow-400">
                      {mentalHealthData.concerning_patterns_count} pattern(s) detected
                    </span>
                  </div>
                  <div className="text-xs text-yellow-300 mt-1">
                    I'm monitoring some emotional patterns that might need attention
                  </div>
                </div>
              )}

              {mentalHealthData.unacknowledged_insights_count > 0 && (
                <div className="bg-blue-500/20 border border-blue-500/30 rounded-lg p-3">
                  <div className="flex items-center space-x-2">
                    <span className="text-blue-400">💡</span>
                    <span className="text-sm font-medium text-blue-400">
                      {mentalHealthData.unacknowledged_insights_count} new insight(s)
                    </span>
                  </div>
                  <div className="text-xs text-blue-300 mt-1">
                    I have some observations about your wellbeing to share
                  </div>
                </div>
              )}
            </motion.div>
          )}

          {/* Beliefs */}
          <motion.div
            className="bg-white/10 rounded-xl p-4"
            whileHover={{ scale: 1.02 }}
          >
            <h3 className="font-semibold mb-3">Evolving Beliefs</h3>
            <div className="space-y-2">
              {cognitiveState.beliefs.length > 0 ? (
                cognitiveState.beliefs.map((belief, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="text-sm text-gray-300 bg-white/10 p-2 rounded"
                  >
                    {belief}
                  </motion.div>
                ))
              ) : (
                <div className="text-gray-400 text-sm">No beliefs formed yet</div>
              )}
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <AppContent />
    </ThemeProvider>
  );
};

export default App;