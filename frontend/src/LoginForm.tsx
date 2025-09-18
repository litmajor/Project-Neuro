import React, { useState } from "react";
import { User, Mail, Lock, Brain } from "lucide-react";

interface LoginFormProps {
  onLogin: (username: string, password: string) => Promise<any>;
  onRegister: (username: string, email: string, password: string) => Promise<any>;
}

const LoginForm: React.FC<LoginFormProps> = ({ onLogin, onRegister }) => {
  const [isLogin, setIsLogin] = useState(true);
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (isLogin) {
        await onLogin(username, password);
      } else {
        await onRegister(username, email, password);
      }
    } catch (error: any) {
      setError(error.message || `${isLogin ? "Login" : "Registration"} failed`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 flex items-center justify-center p-4">
      <div className="bg-white/10 backdrop-blur-xl border border-white/30 rounded-3xl p-8 w-full max-w-md shadow-2xl shadow-blue-500/10 hover:shadow-blue-500/20 transition-all duration-500 animate-[fadeInScale_0.5s_ease-out_forwards]">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-blue-500/20 rounded-full mb-4 animate-[float_4s_ease-in-out_infinite]">
            <Brain className="w-8 h-8 text-blue-400" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent mb-2">
            Welcome to Neuro
          </h1>
          <p className="text-gray-400">
            {isLogin
              ? "Sign in to continue your cognitive journey"
              : "Create your account to begin"}
          </p>
        </div>

        {/* Form */}
        <div className="space-y-6">
          {error && (
            <div className="bg-red-500/20 border border-red-500/30 rounded-lg p-3 text-red-400 text-sm animate-[slideDown_0.3s_ease-out]">
              {error}
            </div>
          )}

          {/* Username */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Username
            </label>
            <div className="relative">
              <User className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent text-white placeholder-gray-400 transition-all duration-300 hover:bg-white/15"
                placeholder="Enter your username"
                required
              />
            </div>
          </div>

          {/* Email (only for registration) */}
          {!isLogin && (
            <div className="animate-[expand_0.3s_ease-out] overflow-hidden">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Email
              </label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
                <input
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent text-white placeholder-gray-400 transition-all duration-300 hover:bg-white/15"
                  placeholder="Enter your email"
                  required={!isLogin}
                />
              </div>
            </div>
          )}

          {/* Password */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-10 pr-4 py-3 bg-white/10 border border-white/20 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent text-white placeholder-gray-400 transition-all duration-300 hover:bg-white/15"
                placeholder="Enter your password"
                required
                minLength={6}
              />
            </div>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmit}
            disabled={loading}
            className="w-full py-4 bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-xl font-semibold text-white transition-all duration-300 shadow-lg hover:shadow-blue-500/25 relative overflow-hidden group hover:scale-105 active:scale-95 transform"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <span className="relative z-10 flex items-center justify-center space-x-2">
              {loading ? (
                <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                <>
                  <span>{isLogin ? "Sign In" : "Create Account"}</span>
                  <span className="animate-[arrow_1.5s_ease-in-out_infinite]">
                    →
                  </span>
                </>
              )}
            </span>
          </button>
        </div>

        {/* Toggle Form Type */}
        <div className="mt-6 text-center">
          <p className="text-gray-400">
            {isLogin ? "Don't have an account?" : "Already have an account?"}
          </p>
          <button
            onClick={() => {
              setIsLogin(!isLogin);
              setError("");
            }}
            className="mt-2 text-blue-400 hover:text-blue-300 font-medium transition-colors"
          >
            {isLogin ? "Create one here" : "Sign in here"}
          </button>
        </div>
      </div>

      
    </div>
  );
};

export default LoginForm;