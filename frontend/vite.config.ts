import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    allowedHosts: [
      'localhost',
      '.replit.dev',
      'e647f0bf-1df9-4bc5-a1fc-ff848945fbe5-00-gkmd3rwmaweh.picard.replit.dev'
    ]
  }
})
