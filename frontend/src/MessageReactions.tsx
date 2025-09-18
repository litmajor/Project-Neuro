
import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ThumbsUp, ThumbsDown, Heart, Smile, Frown, Meh } from 'lucide-react';

interface MessageReactionsProps {
  messageId: string;
  onReaction: (messageId: string, reaction: string) => void;
  existingReactions?: Record<string, number>;
}

const REACTIONS = [
  { icon: ThumbsUp, name: 'like', color: 'text-green-400' },
  { icon: ThumbsDown, name: 'dislike', color: 'text-red-400' },
  { icon: Heart, name: 'love', color: 'text-pink-400' },
  { icon: Smile, name: 'happy', color: 'text-yellow-400' },
  { icon: Meh, name: 'neutral', color: 'text-gray-400' },
  { icon: Frown, name: 'sad', color: 'text-blue-400' }
];

export const MessageReactions: React.FC<MessageReactionsProps> = ({
  messageId,
  onReaction,
  existingReactions = {}
}) => {
  const [showReactions, setShowReactions] = useState(false);

  const handleReaction = (reactionName: string) => {
    onReaction(messageId, reactionName);
    setShowReactions(false);
  };

  return (
    <div className="relative">
      <motion.button
        className="text-xs text-gray-400 hover:text-gray-200 transition-colors"
        onClick={() => setShowReactions(!showReactions)}
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
      >
        React
      </motion.button>

      <AnimatePresence>
        {showReactions && (
          <motion.div
            initial={{ opacity: 0, scale: 0.8, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.8, y: 10 }}
            className="absolute bottom-full mb-2 left-0 flex space-x-2 bg-gray-800/90 backdrop-blur-md rounded-lg p-2 border border-gray-600"
          >
            {REACTIONS.map(({ icon: Icon, name, color }) => (
              <motion.button
                key={name}
                className={`p-2 hover:bg-white/10 rounded-lg transition-colors ${color}`}
                onClick={() => handleReaction(name)}
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <Icon className="w-4 h-4" />
              </motion.button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Display existing reactions */}
      {Object.entries(existingReactions).length > 0 && (
        <div className="flex space-x-1 mt-2">
          {Object.entries(existingReactions).map(([reaction, count]) => {
            const reactionConfig = REACTIONS.find(r => r.name === reaction);
            if (!reactionConfig || count === 0) return null;
            
            const Icon = reactionConfig.icon;
            return (
              <span
                key={reaction}
                className={`text-xs px-2 py-1 rounded-full bg-gray-700/50 ${reactionConfig.color} flex items-center space-x-1`}
              >
                <Icon className="w-3 h-3" />
                <span>{count}</span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
};
