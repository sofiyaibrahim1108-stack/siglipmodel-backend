// siglip.js — calls Python model service for embeddings
// No model loaded here — Python handles all AI inference

import fetch from "node-fetch";

// Python model service URL
const MODEL_SERVICE = process.env.MODEL_SERVICE_URL || "http://localhost:7000";

export async function loadModels() {
  // Just verify Python service is running
  console.log("🔗 Connecting to Python model service...");
  try {
    const res  = await fetch(`${MODEL_SERVICE}/health`);
    const data = await res.json();
    console.log(`✨ Model service ready — ${data.model} on ${data.device}`);
  } catch {
    console.error("❌ Python model service not running! Start it with: python model_service.py");
    process.exit(1);
  }
}

export async function getImageEmbedding(imagePath) {
  // Send image file to Python service → get back embedding
  const { FormData, Blob } = await import("node-fetch");
  const fs   = await import("fs");

  const form = new FormData();
  const fileBuffer = fs.readFileSync(imagePath);
  const blob = new Blob([fileBuffer]);
  form.append("image", blob, imagePath.split(/[\\/]/).pop());

  const res  = await fetch(`${MODEL_SERVICE}/embed/image`, {
    method: "POST",
    body:   form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(`Model service error: ${err.error}`);
  }

  const data = await res.json();
  return data.embedding; // already L2-normalized float[]
}

export async function getTextEmbedding(text) {
  // Send text to Python service → get back embedding
  const form = new URLSearchParams();
  form.append("text", text);

  const res = await fetch(`${MODEL_SERVICE}/embed/text`, {
    method:  "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body:    form,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(`Model service error: ${err.error}`);
  }

  const data = await res.json();
  return data.embedding; // already L2-normalized float[]
}