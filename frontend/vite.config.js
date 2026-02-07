import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/chat': { target: 'http://localhost:8000', changeOrigin: true },
      '/health': { target: 'http://localhost:8000', changeOrigin: true },
      '/initialize': { target: 'http://localhost:8000', changeOrigin: true },
      '/analyze': { target: 'http://localhost:8000', changeOrigin: true },
      '/items': { target: 'http://localhost:8000', changeOrigin: true },
      '/recommendations': { target: 'http://localhost:8000', changeOrigin: true },
      '/pricing-suggestions': { target: 'http://localhost:8000', changeOrigin: true },
      '/ask': { target: 'http://localhost:8000', changeOrigin: true },
      '/export': { target: 'http://localhost:8000', changeOrigin: true },
      '/restaurants': { target: 'http://localhost:8000', changeOrigin: true },
      '/menu-items': { target: 'http://localhost:8000', changeOrigin: true },
      '/orders': { target: 'http://localhost:8000', changeOrigin: true },
      '/inventory': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
})
