import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Static, relative-base build so it can be hosted on GitHub Pages / any CDN.
export default defineConfig({
  base: "./",
  plugins: [react()],
  server: { port: 5173 },
  preview: { port: 4173 },
});
