import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const apiPort = process.env.API_PORT ?? '8000'
const apiOrigin = `http://localhost:${apiPort}`

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/search': apiOrigin,
      '/images': apiOrigin,
    },
  },
})
