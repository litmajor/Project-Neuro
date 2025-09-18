
import React from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Calendar, 
  Heart, 
  Zap, 
  MessageSquare, 
  FileText, 
  Mic, 
  Users,
  Shield,
  Sparkles,
  ArrowRight,
  CheckCircle,
  Star
} from 'lucide-react';

interface LandingPageProps {
  onGetStarted: () => void;
}

const LandingPage: React.FC<LandingPageProps> = ({ onGetStarted }) => {
  const features = [
    {
      icon: Brain,
      title: "Persistent Memory",
      description: "Never forget important details. Neuro remembers every conversation and builds a comprehensive understanding of your preferences.",
      color: "text-purple-400"
    },
    {
      icon: Heart,
      title: "Emotional Intelligence",
      description: "Advanced mood tracking and empathetic responses. Neuro adapts its communication style based on your emotional state.",
      color: "text-pink-400"
    },
    {
      icon: Calendar,
      title: "Smart Scheduling",
      description: "Intelligent calendar management with mood-aware scheduling. Get personalized advice on when to tackle different tasks.",
      color: "text-blue-400"
    },
    {
      icon: FileText,
      title: "Document Intelligence",
      description: "Upload and analyze documents, images, and files. Neuro provides contextual insights and helps organize your information.",
      color: "text-green-400"
    },
    {
      icon: Mic,
      title: "Voice Integration",
      description: "Natural voice conversations with speech-to-text and text-to-speech capabilities. Interact hands-free with your assistant.",
      color: "text-yellow-400"
    },
    {
      icon: Users,
      title: "Collaborative Spaces",
      description: "Share conversations and collaborate with others. Perfect for team projects and shared decision-making.",
      color: "text-indigo-400"
    }
  ];

  const useCases = [
    {
      title: "Daily Planning",
      description: "Get personalized schedules based on your energy levels and mood patterns",
      icon: "📅"
    },
    {
      title: "Emotional Support",
      description: "Receive empathetic responses and mood-based advice for better mental wellness",
      icon: "💝"
    },
    {
      title: "Task Management",
      description: "Smart reminders and task prioritization that adapts to your working style",
      icon: "✅"
    },
    {
      title: "Knowledge Management",
      description: "Organize and retrieve information from documents, notes, and conversations",
      icon: "📚"
    },
    {
      title: "Decision Making",
      description: "Get thoughtful analysis and recommendations for important choices",
      icon: "🎯"
    },
    {
      title: "Creative Projects",
      description: "Brainstorming partner that remembers your ideas and helps develop them",
      icon: "🎨"
    }
  ];

  const testimonials = [
    {
      name: "Sarah Chen",
      role: "Product Manager",
      content: "Neuro has transformed how I manage my daily tasks. It actually remembers my preferences and adapts to my mood!",
      rating: 5
    },
    {
      name: "Marcus Rodriguez",
      role: "Creative Director",
      content: "The emotional intelligence is incredible. It's like having a thoughtful friend who never forgets anything.",
      rating: 5
    },
    {
      name: "Dr. Emily Watson",
      role: "Researcher",
      content: "Finally, an AI assistant that learns and evolves. The persistent memory is a game-changer for my research work.",
      rating: 5
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 text-white overflow-x-hidden">
      {/* Hero Section */}
      <motion.section 
        className="relative min-h-screen flex items-center justify-center px-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
      >
        <div className="absolute inset-0 bg-gradient-to-b from-transparent to-black/20" />
        <div className="max-w-6xl mx-auto text-center z-10">
          <motion.div
            initial={{ y: -50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="mb-8"
          >
            <div className="inline-flex items-center justify-center w-20 h-20 bg-blue-500/20 rounded-full mb-6">
              <Brain className="w-12 h-12 text-blue-400" />
            </div>
            <h1 className="text-6xl md:text-8xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-6">
              Neuro Assistant
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-3xl mx-auto">
              Your Persistent AI Companion with True Memory, Emotional Intelligence, and Adaptive Learning
            </p>
          </motion.div>

          <motion.div
            initial={{ y: 50, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-12"
          >
            <motion.button
              onClick={onGetStarted}
              className="px-8 py-4 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 rounded-full font-semibold text-lg transition-all duration-300 shadow-lg hover:shadow-blue-500/25 flex items-center space-x-2"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              <span>Get Started Free</span>
              <ArrowRight className="w-5 h-5" />
            </motion.button>
            <motion.button
              className="px-8 py-4 border-2 border-white/20 hover:border-white/40 rounded-full font-semibold text-lg transition-all duration-300"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Watch Demo
            </motion.button>
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.6 }}
            className="flex flex-wrap justify-center gap-8 text-sm text-gray-400"
          >
            <div className="flex items-center space-x-2">
              <Shield className="w-4 h-4 text-green-400" />
              <span>Privacy First</span>
            </div>
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              <span>AI-Powered</span>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircle className="w-4 h-4 text-blue-400" />
              <span>No Setup Required</span>
            </div>
          </motion.div>
        </div>
      </motion.section>

      {/* Features Section */}
      <motion.section 
        className="py-20 px-4"
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
      >
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Powerful Features That
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent"> Adapt to You</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Unlike traditional assistants that forget everything, Neuro builds a lasting relationship with you through advanced cognitive capabilities.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6 hover:bg-white/10 transition-all duration-300 group"
                whileHover={{ scale: 1.02, y: -5 }}
              >
                <div className={`inline-flex p-3 bg-gradient-to-br from-gray-700 to-gray-800 rounded-xl mb-4 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className={`w-6 h-6 ${feature.color}`} />
                </div>
                <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
                <p className="text-gray-300 leading-relaxed">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Use Cases Section */}
      <motion.section 
        className="py-20 px-4 bg-black/20"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
      >
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Perfect for Every
              <span className="bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent"> Aspect of Life</span>
            </h2>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              From personal productivity to creative projects, Neuro adapts to your unique needs and grows with you.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {useCases.map((useCase, index) => (
              <motion.div
                key={useCase.title}
                initial={{ opacity: 0, scale: 0.9 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="bg-gradient-to-br from-white/5 to-white/10 backdrop-blur-md border border-white/10 rounded-xl p-6 hover:border-white/20 transition-all duration-300"
                whileHover={{ y: -5 }}
              >
                <div className="text-3xl mb-4">{useCase.icon}</div>
                <h3 className="text-lg font-semibold mb-2">{useCase.title}</h3>
                <p className="text-gray-300 text-sm">{useCase.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* Testimonials Section */}
      <motion.section 
        className="py-20 px-4"
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
      >
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Loved by
              <span className="bg-gradient-to-r from-yellow-400 to-orange-400 bg-clip-text text-transparent"> Thousands</span>
            </h2>
            <p className="text-xl text-gray-300">
              See what our users say about their experience with Neuro
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {testimonials.map((testimonial, index) => (
              <motion.div
                key={testimonial.name}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.2 }}
                viewport={{ once: true }}
                className="bg-white/5 backdrop-blur-md border border-white/10 rounded-2xl p-6"
              >
                <div className="flex mb-4">
                  {[...Array(testimonial.rating)].map((_, i) => (
                    <Star key={i} className="w-5 h-5 text-yellow-400 fill-current" />
                  ))}
                </div>
                <p className="text-gray-300 mb-6 italic">"{testimonial.content}"</p>
                <div>
                  <div className="font-semibold">{testimonial.name}</div>
                  <div className="text-sm text-gray-400">{testimonial.role}</div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.section>

      {/* CTA Section */}
      <motion.section 
        className="py-20 px-4 bg-gradient-to-r from-blue-900/30 to-purple-900/30"
        initial={{ opacity: 0, scale: 0.95 }}
        whileInView={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.8 }}
        viewport={{ once: true }}
      >
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-6xl font-bold mb-6">
            Ready to Meet Your
            <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent"> Perfect AI Companion?</span>
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Join thousands who have already discovered the power of persistent AI assistance.
          </p>
          <motion.button
            onClick={onGetStarted}
            className="px-10 py-5 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 rounded-full font-semibold text-xl transition-all duration-300 shadow-2xl hover:shadow-blue-500/25 flex items-center space-x-3 mx-auto"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <span>Start Your Journey</span>
            <Sparkles className="w-6 h-6" />
          </motion.button>
        </div>
      </motion.section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-white/10">
        <div className="max-w-7xl mx-auto text-center">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <Brain className="w-6 h-6 text-blue-400" />
            <span className="text-lg font-semibold">Neuro Assistant</span>
          </div>
          <p className="text-gray-400">
            © 2025 Neuro Assistant. Built with 💜 for better human-AI collaboration.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
