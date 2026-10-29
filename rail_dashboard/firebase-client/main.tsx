import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "../app/globals.css";
import { FirebaseDashboard } from "./FirebaseDashboard";

const root = document.getElementById("root");
const canonicalFirebaseHost = "your-firebase-project.firebaseapp.com";

if (!root) {
  throw new Error("Dashboard root element was not found.");
}

if (window.location.hostname === "your-firebase-project.web.app") {
  const canonicalUrl = new URL(window.location.href);
  canonicalUrl.hostname = canonicalFirebaseHost;
  window.location.replace(canonicalUrl.toString());
} else {
  createRoot(root).render(
    <StrictMode>
      <FirebaseDashboard />
    </StrictMode>,
  );
}
