import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 开发时把 /api 代理到 FastAPI 后端（subtitle-translator serve 默认端口）
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:7860",
        changeOrigin: true,
      },
    },
  },
});
