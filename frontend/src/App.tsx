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
import { LoginForm } from './LoginForm';
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

  // Refs
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const currentStreamingMessage = useRef<string>('');

  // WebSocket connection for real-time updates
  useEffect(() => {
    const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const apiBase = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host.replace(':5173', ':8000');
    const wsUrl = `ws://${apiBase}/ws/${clientId}`;

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
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Send message to AI with streaming response
  const sendMessage = async () => {
    if (!inputValue.trim() || isStreaming) return;

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
      const apiBase = window.location.hostname === 'localhost' ? 'localhost:8000' : window.location.host.replace(':5173', ':8000');
      const response = await fetch(`http://${apiBase}/api/chat/stream`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(isAuthenticated ? { 'Authorization': `Bearer ${localStorage.getItem('auth_token')}` } : {})
        },
        body: JSON.stringify({
          message: userMessage.content,
          conversation_history: messages.slice(-10), // Last 10 messages for context
          cognitive_context: cognitiveState
        })
      });

      if (!response.ok) {
        throw new Error('Failed to get AI response');
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
    } catch (error) {
      console.error('Error sending message:', error);

      // Update message with error
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessageId 
          ? { 
              ...msg, 
              content: '❌ Sorry, I encountered an error. Please try again.',
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

  // Speak assistant messages when voice is enabled
  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      if (lastMessage.role === 'assistant' && !lastMessage.isStreaming && (window as any).speakText) {
        // Clean up the message content for speech
        const cleanContent = lastMessage.content.replace(/[*#`]/g, '').replace(/\n+/g, ' ');
        (window as any).speakText(cleanContent);
      }
    }
  }, [messages]);

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

  // Show login form if not authenticated
  if (!isAuthenticated) {
    return <LoginForm onLogin={login} onRegister={register} />;
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
                Neuro v3
              </h1>
              <p className={`text-sm ${theme === 'light' ? 'text-gray-600' : 'text-gray-400'}`}>
                Advanced Cognitive Agent
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
              {isConnected ? 'CONNECTED' : 'DISCONNECTED'}
            </motion.div>

            {/* Collaboration Button */}
            <motion.button
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              onClick={() => setShowCollaborations(!showCollaborations)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Users className="w-5 h-5" />
            </motion.button>

            {/* File Upload Button */}
            <motion.button
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              onClick={() => setShowFileUpload(!showFileUpload)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Sparkles className="w-5 h-5" />
            </motion.button>

            {/* Settings Button */}
            <motion.button
              className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              onClick={() => setShowSettings(!showSettings)}
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <Settings className="w-5 h-5" />
            </motion.button>

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
                          className="flex space-x-1"
                          animate={{ opacity: [0.5, 1, 0.5] }}
                          transition={{ duration: 1, repeat: Infinity }}
                        >
                          <div className="w-1 h-1 bg-blue-400 rounded-full" />
                          <div className="w-1 h-1 bg-blue-400 rounded-full" />
                          <div className="w-1 h-1 bg-blue-400 rounded-full" />
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

          {/* File Upload Section */}
          <AnimatePresence>
            {showFileUpload && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="p-6 bg-white/5 backdrop-blur-md border-t border-white/20"
              >
                <FileUpload 
                  onFileUploaded={handleFileUploaded}
                  conversationId={currentSharedConversation ? undefined : 1}
                  sharedConversationId={currentSharedConversation || undefined}
                />
              </motion.div>
            )}
          </AnimatePresence>

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
          {/* Collaborations Section */}
          <AnimatePresence>
            {showCollaborations && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mb-6"
              >
                <SharedConversations onJoinConversation={handleJoinSharedConversation} />
              </motion.div>
            )}
          </AnimatePresence>
          <h2 className="text-xl font-bold mb-6 flex items-center space-x-2">
            <Activity className="w-5 h-5 text-blue-400" />
            <span>Cognitive State</span>
          </h2>

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