import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd())
  const apiUrl = env.VITE_API_URL || 'http://localhost:8000'

  return {
    plugins: [react()],
    server: {
      proxy: {
        '/chat': { target: apiUrl, changeOrigin: true },
        '/health': { target: apiUrl, changeOrigin: true },
        '/initialize': { target: apiUrl, changeOrigin: true },
        '/analyze': { target: apiUrl, changeOrigin: true },
        '/items': { target: apiUrl, changeOrigin: true },
        '/recommendations': { target: apiUrl, changeOrigin: true },
        '/pricing-suggestions': { target: apiUrl, changeOrigin: true },
        '/ask': { target: apiUrl, changeOrigin: true },
        '/export': { target: apiUrl, changeOrigin: true },
        '/restaurants': { target: apiUrl, changeOrigin: true },
        '/menu-items': { target: apiUrl, changeOrigin: true },
        '/orders': { target: apiUrl, changeOrigin: true },
        '/inventory': { target: apiUrl, changeOrigin: true },
      },
    },
  }
})
