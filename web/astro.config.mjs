import { defineConfig } from "astro/config";

export default defineConfig({
  devToolbar: { enabled: false },
  server: { port: 4321, host: "127.0.0.1" },
  vite: {
    server: {
      proxy: {
        "/v1": { target: "http://127.0.0.1:8000", changeOrigin: true },
        "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
        "/authenticate": { target: "http://127.0.0.1:8000", changeOrigin: true },
        "/admin": { target: "http://127.0.0.1:8000", changeOrigin: true },
      },
    },
  },
});
