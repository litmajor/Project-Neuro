import { useState, useEffect } from 'react';

export interface User {
  id: number;
  username: string;
  email: string;
  created_at: string;
  ai_personality: Record<string, number>;
  preferred_mood: string;
  energy_preference: number;
}

export interface AuthToken {
  access_token: string;
  token_type: string;
  user: User;
}

const isLocal = window.location.hostname === 'localhost';
const host = window.location.host;
// For Replit: if we're on a Replit domain, construct the backend URL properly
const API_BASE = isLocal 
  ? 'localhost:8000' 
  : host.includes('replit.dev') || host.includes('replit.app')
    ? host.replace(/^[^-]+-/, '8000-') // Replace the port prefix
    : 'localhost:8000';
const API_PROTOCOL = isLocal ? 'http' : 'https';

export class AuthService {
  private token: string | null = null;
  private user: User | null = null;

  constructor() {
    // Load token from localStorage on initialization
    const savedToken = localStorage.getItem('auth_token');
    const savedUser = localStorage.getItem('user_data');

    if (savedToken && savedUser) {
      this.token = savedToken;
      this.user = JSON.parse(savedUser);
    }
  }

  async register(username: string, email: string, password: string): Promise<AuthToken> {
    const url = `${API_PROTOCOL}://${API_BASE}/api/auth/register`;
    console.log('🔗 Registration URL:', url);
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, email, password }),
      });

      console.log('📡 Response status:', response.status);
      console.log('📡 Response headers:', Object.fromEntries(response.headers.entries()));

      if (!response.ok) {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const error = await response.json();
          throw new Error(error.detail || 'Registration failed');
        } else {
          const text = await response.text();
          console.error('❌ Non-JSON error response:', text);
          throw new Error(`Registration failed: ${response.status} ${response.statusText}`);
        }
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        console.error('❌ Expected JSON response but got:', text);
        throw new Error('Invalid response format from server');
      }

      const authData: AuthToken = await response.json();
      this.setAuth(authData);
      return authData;
    } catch (error) {
      console.error('❌ Registration error:', error);
      throw error;
    }
  }

  async login(username: string, password: string): Promise<AuthToken> {
    const url = `${API_PROTOCOL}://${API_BASE}/api/auth/login`;
    console.log('🔗 Login URL:', url);
    
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ username, password }),
      });

      console.log('📡 Response status:', response.status);

      if (!response.ok) {
        const contentType = response.headers.get('content-type');
        if (contentType && contentType.includes('application/json')) {
          const error = await response.json();
          throw new Error(error.detail || 'Login failed');
        } else {
          const text = await response.text();
          console.error('❌ Non-JSON error response:', text);
          throw new Error(`Login failed: ${response.status} ${response.statusText}`);
        }
      }

      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        console.error('❌ Expected JSON response but got:', text);
        throw new Error('Invalid response format from server');
      }

      const authData: AuthToken = await response.json();
      this.setAuth(authData);
      return authData;
    } catch (error) {
      console.error('❌ Login error:', error);
      throw error;
    }
  }

  async updatePersonality(personality: Record<string, number>, mood: string, energy: number): Promise<User> {
    const response = await fetch(`${API_PROTOCOL}://${API_BASE}/api/auth/personality`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.token}`,
      },
      body: JSON.stringify({
        ai_personality: personality,
        preferred_mood: mood,
        energy_preference: energy,
      }),
    });

    if (!response.ok) {
      throw new Error('Failed to update personality settings');
    }

    const user: User = await response.json();
    this.user = user;
    localStorage.setItem('user_data', JSON.stringify(user));
    return user;
  }

  logout(): void {
    this.token = null;
    this.user = null;
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_data');
  }

  private setAuth(authData: AuthToken): void {
    this.token = authData.access_token;
    this.user = authData.user;
    localStorage.setItem('auth_token', authData.access_token);
    localStorage.setItem('user_data', JSON.stringify(authData.user));
  }

  getToken(): string | null {
    return this.token;
  }

  getUser(): User | null {
    return this.user;
  }

  isAuthenticated(): boolean {
    return this.token !== null;
  }

  getAuthHeaders(): Record<string, string> {
    return this.token ? {
      'Authorization': `Bearer ${this.token}`,
      'Content-Type': 'application/json',
    } : {
      'Content-Type': 'application/json',
    };
  }
}

export const authService = new AuthService();

export function useAuth() {
  const [user, setUser] = useState<User | null>(authService.getUser());
  const [isAuthenticated, setIsAuthenticated] = useState(authService.isAuthenticated());

  const login = async (username: string, password: string) => {
    try {
      const authData = await authService.login(username, password);
      setUser(authData.user);
      setIsAuthenticated(true);
      return authData;
    } catch (error) {
      console.error('Login failed:', error);
      throw error;
    }
  };

  const register = async (username: string, email: string, password: string) => {
    try {
      const authData = await authService.register(username, email, password);
      setUser(authData.user);
      setIsAuthenticated(true);
      return authData;
    } catch (error) {
      console.error('Registration failed:', error);
      throw error;
    }
  };

  const logout = () => {
    authService.logout();
    setUser(null);
    setIsAuthenticated(false);
  };

  const updatePersonality = async (personality: Record<string, number>, mood: string, energy: number): Promise<User> => {
    try {
      const updatedUser = await authService.updatePersonality(personality, mood, energy);
      setUser(updatedUser);
      return updatedUser;
    } catch (error) {
      console.error('Failed to update personality:', error);
      throw error;
    }
  };

  const joinSharedConversation = async (conversationId: number) => {
    // This would be called when user joins a shared conversation
    // to set up WebSocket listeners for that specific conversation
    console.log(`Joining shared conversation ${conversationId}`);
  };

  return {
    user,
    isAuthenticated,
    login,
    register,
    logout,
    updatePersonality,
    joinSharedConversation
  };
}