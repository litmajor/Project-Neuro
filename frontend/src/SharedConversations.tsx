
import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Users, 
  Plus, 
  MessageCircle, 
  Eye, 
  Crown,
  User as UserIcon,
  Send,
  MoreVertical
} from 'lucide-react';

interface SharedConversation {
  id: number;
  name: string;
  description?: string;
  participant_count: number;
  online_count: number;
  last_activity: string;
  role: string;
}

interface SharedConversationsProps {
  onJoinConversation: (conversationId: number) => void;
}

export const SharedConversations: React.FC<SharedConversationsProps> = ({ onJoinConversation }) => {
  const [conversations, setConversations] = useState<SharedConversation[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newConversation, setNewConversation] = useState({
    name: '',
    description: '',
    participant_usernames: ''
  });
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchSharedConversations();
  }, []);

  const fetchSharedConversations = async () => {
    try {
      const response = await fetch('/api/shared-conversations', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        setConversations(data);
      }
    } catch (error) {
      console.error('Failed to fetch shared conversations:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const createSharedConversation = async () => {
    try {
      const response = await fetch('/api/shared-conversations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('auth_token')}`
        },
        body: JSON.stringify({
          name: newConversation.name,
          description: newConversation.description,
          participant_usernames: newConversation.participant_usernames
            .split(',')
            .map(u => u.trim())
            .filter(u => u)
        })
      });

      if (response.ok) {
        setShowCreateForm(false);
        setNewConversation({ name: '', description: '', participant_usernames: '' });
        fetchSharedConversations();
      }
    } catch (error) {
      console.error('Failed to create shared conversation:', error);
    }
  };

  const formatTimeAgo = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInMinutes = Math.floor((now.getTime() - date.getTime()) / 60000);
    
    if (diffInMinutes < 1) return 'Just now';
    if (diffInMinutes < 60) return `${diffInMinutes}m ago`;
    if (diffInMinutes < 1440) return `${Math.floor(diffInMinutes / 60)}h ago`;
    return `${Math.floor(diffInMinutes / 1440)}d ago`;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
        >
          <Users className="w-8 h-8 text-blue-400" />
        </motion.div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold flex items-center space-x-2">
          <Users className="w-5 h-5 text-blue-400" />
          <span>Team Workspaces</span>
        </h2>
        <motion.button
          onClick={() => setShowCreateForm(true)}
          className="p-2 bg-blue-500 hover:bg-blue-600 rounded-lg transition-colors"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          <Plus className="w-5 h-5" />
        </motion.button>
      </div>

      {/* Create Form */}
      <AnimatePresence>
        {showCreateForm && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-white/10 rounded-xl p-4 space-y-4"
          >
            <h3 className="font-semibold text-lg">Create Team Workspace</h3>
            <input
              type="text"
              placeholder="Workspace name..."
              value={newConversation.name}
              onChange={(e) => setNewConversation(prev => ({ ...prev, name: e.target.value }))}
              className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400"
            />
            <textarea
              placeholder="Description (optional)..."
              value={newConversation.description}
              onChange={(e) => setNewConversation(prev => ({ ...prev, description: e.target.value }))}
              className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 resize-none"
              rows={2}
            />
            <input
              type="text"
              placeholder="Invite users (comma-separated usernames)..."
              value={newConversation.participant_usernames}
              onChange={(e) => setNewConversation(prev => ({ ...prev, participant_usernames: e.target.value }))}
              className="w-full p-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400"
            />
            <div className="flex space-x-2">
              <motion.button
                onClick={createSharedConversation}
                disabled={!newConversation.name}
                className="flex-1 p-3 bg-blue-500 hover:bg-blue-600 disabled:opacity-50 rounded-lg transition-colors"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                Create Workspace
              </motion.button>
              <button
                onClick={() => setShowCreateForm(false)}
                className="px-4 py-3 bg-gray-500 hover:bg-gray-600 rounded-lg transition-colors"
              >
                Cancel
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Conversations List */}
      <div className="space-y-3">
        {conversations.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <Users className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No team workspaces yet.</p>
            <p className="text-sm">Create one to start collaborating!</p>
          </div>
        ) : (
          conversations.map((conv) => (
            <motion.div
              key={conv.id}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-white/10 rounded-xl p-4 hover:bg-white/15 transition-colors cursor-pointer"
              onClick={() => onJoinConversation(conv.id)}
              whileHover={{ scale: 1.02 }}
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-2 mb-2">
                    <h3 className="font-semibold text-lg">{conv.name}</h3>
                    {conv.role === 'owner' && (
                      <Crown className="w-4 h-4 text-yellow-400" />
                    )}
                  </div>
                  
                  {conv.description && (
                    <p className="text-gray-400 text-sm mb-3">{conv.description}</p>
                  )}

                  <div className="flex items-center space-x-4 text-sm text-gray-400">
                    <div className="flex items-center space-x-1">
                      <UserIcon className="w-4 h-4" />
                      <span>{conv.participant_count} members</span>
                    </div>
                    
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-400 rounded-full" />
                      <span>{conv.online_count} online</span>
                    </div>

                    <div className="flex items-center space-x-1">
                      <MessageCircle className="w-4 h-4" />
                      <span>{formatTimeAgo(conv.last_activity)}</span>
                    </div>
                  </div>
                </div>

                <button className="p-2 hover:bg-white/10 rounded-lg">
                  <MoreVertical className="w-4 h-4" />
                </button>
              </div>
            </motion.div>
          ))
        )}
      </div>
    </div>
  );
};
