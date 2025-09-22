// Drizzle ORM schema for Project-Neuro
// Add your tables and relations here
import { pgTable, serial, varchar, text, timestamp, integer, boolean, json } from 'drizzle-orm/pg-core';

export const users = pgTable('users', {
  id: serial('id').primaryKey(),
  username: varchar('username', { length: 64 }).notNull(),
  email: varchar('email', { length: 128 }).notNull(),
  hashed_password: varchar('hashed_password', { length: 128 }).notNull(),
  ai_personality: json('ai_personality'),
  preferred_mood: varchar('preferred_mood', { length: 32 }),
  energy_preference: integer('energy_preference'),
  created_at: timestamp('created_at').defaultNow().notNull(),
});

export const conversations = pgTable('conversations', {
  id: serial('id').primaryKey(),
  user_id: integer('user_id').notNull(),
  title: varchar('title', { length: 128 }),
  created_at: timestamp('created_at').defaultNow().notNull(),
  updated_at: timestamp('updated_at').defaultNow().notNull(),
});

export const messages = pgTable('messages', {
  id: serial('id').primaryKey(),
  conversation_id: integer('conversation_id').notNull(),
  role: varchar('role', { length: 32 }).notNull(),
  content: text('content').notNull(),
  timestamp: timestamp('timestamp').defaultNow().notNull(),
  cognitive_state: json('cognitive_state'),
});

export const memories = pgTable('memories', {
  id: serial('id').primaryKey(),
  user_id: integer('user_id').notNull(),
  content: text('content').notNull(),
  category: varchar('category', { length: 32 }),
  importance_score: integer('importance_score'),
  created_at: timestamp('created_at').defaultNow().notNull(),
  accessed_count: integer('accessed_count').default(0),
});

export const user_preferences = pgTable('user_preferences', {
  id: serial('id').primaryKey(),
  user_id: integer('user_id').notNull(),
  preference_type: varchar('preference_type', { length: 32 }).notNull(),
  preference_value: varchar('preference_value', { length: 128 }).notNull(),
  confidence_score: integer('confidence_score'),
  learned_from: varchar('learned_from', { length: 64 }),
  updated_at: timestamp('updated_at').defaultNow().notNull(),
});
