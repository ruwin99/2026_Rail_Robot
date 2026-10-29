import { readFile } from "node:fs/promises";
import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const safePublicAssets = [
  ["./public/dolphin-mark.png", "dolphin-mark.png"],
  ["./public/demo/rust-web-demo.jpg", "demo/rust-web-demo.jpg"],
  ["./public/demo/crack-web-demo.jpg", "demo/crack-web-demo.jpg"],
] as const;

export default defineConfig({
  root: fileURLToPath(new URL("./firebase-client", import.meta.url)),
  publicDir: false,
  envDir: fileURLToPath(new URL(".", import.meta.url)),
  plugins: [
    react(),
    {
      name: "firebase-safe-public-assets",
      async buildStart() {
        for (const [source, fileName] of safePublicAssets) {
          this.emitFile({
            type: "asset",
            fileName,
            source: await readFile(fileURLToPath(new URL(source, import.meta.url))),
          });
        }
      },
    },
  ],
  build: {
    outDir: fileURLToPath(new URL("./firebase-dist", import.meta.url)),
    emptyOutDir: true,
  },
});
