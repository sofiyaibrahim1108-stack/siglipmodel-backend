// models/siglip.js
// calls Python model service for embeddings
// No model loaded here — Python handles all AI inference

import fetch from "node-fetch";
import fs from "fs";

const MODEL_SERVICE = process.env.MODEL_SERVICE_URL || "http://localhost:8000";

export async function loadModels() {
  console.log("🔗 Connecting to Python model service...");
  try {
    const res  = await fetch(`${MODEL_SERVICE}/health`);
    const data = await res.json();
    console.log(`✨ Model service ready — ${data.siglip_model || "SigLIP2"} on ${data.device || "cpu"}`);
  } catch {
    console.warn("⚠️  Python model service not reachable yet. Start it with: python modelService.py");
    console.warn("   Node.js will continue — embedding calls will fail until Python is running.");
  }
}

export async function getImageEmbedding(imagePath) {
  try {
    if (!fs.existsSync(imagePath)) {
      throw new Error(`File does not exist on disk path: ${imagePath}`);
    }

    // Direct binary buffer reading
    const fileBuffer = fs.readFileSync(imagePath);
    const filename = imagePath.split(/[\\/]/).pop();

    // Query string parameter structure mapping with clean raw headers payload configurations
    const res = await fetch(`${MODEL_SERVICE}/embed/image?filename=${encodeURIComponent(filename)}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream"
      },
      body: fileBuffer
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.error || "Failed to retrieve image vectors embedding data representation matrix");
    }

    const data = await res.json();
    return data.embedding;
  } catch (error) {
    console.error("❌ Image Embedding Client Error:", error.message);
    throw error;
  }
}

export async function getTextEmbedding(text) {
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
  return data.embedding;
}