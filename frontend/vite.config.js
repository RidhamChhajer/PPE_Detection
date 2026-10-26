import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";
export default defineConfig({
  base: "/static/",
  plugins: [react()],
  build: { outDir: "../app/static", emptyOutDir: true },
  server: {
    host: "127.0.0.1",
    port: 5174,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8876",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (req) =>
            req.setHeader("origin", "http://127.0.0.1:8876"),
          );
        },
      },
    },
  },
});
