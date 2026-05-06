import express from "express";
import multer from "multer";
import cors from "cors";
import dotenv from "dotenv";
import { MongoClient } from "mongodb";
import {
  AutoProcessor,
  AutoTokenizer,
  SiglipVisionModel,
  SiglipTextModel,
  RawImage,
  env,
} from "@huggingface/transformers";
import path from "path";
import fs from "fs";

dotenv.config();

// ─── ENV CONFIG ──────────────────────────────────────────────────────────────
// transformers.js auto-caches to ~/.cache/huggingface/hub
// Already-downloaded files are reused — only missing files are fetched
env.allowLocalModels = true;
env.allowRemoteModels = true;

const app = express();
app.use(cors());
app.use(express.json());
app.use("/uploads", express.static("uploads"));

// Ensure uploads folder exists
if (!fs.existsSync("uploads")) fs.mkdirSync("uploads");

// ─── MULTER STORAGE ──────────────────────────────────────────────────────────
const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    const unique = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
    cb(null, unique + path.extname(file.originalname));
  },
});
const upload = multer({ storage });

// ─── SIGLIP2 MODEL ───────────────────────────────────────────────────────────
// Single unified model for BOTH image and text embeddings in one shared space
const MODEL_ID = "onnx-community/siglip2-base-patch16-224-ONNX";

// ─── MONGODB ─────────────────────────────────────────────────────────────────
const client = new MongoClient(process.env.MONGO_URI);
let collection;

async function connectDB() {
  try {
    await client.connect();
    const db = client.db(process.env.DB_NAME);
    collection = db.collection(process.env.COLLECTION_NAME);
    console.log("✅ MongoDB Connected");
  } catch (err) {
    console.error("❌ MongoDB Connection Error:", err);
    process.exit(1);
  }
}

// ─── MODEL INSTANCES ─────────────────────────────────────────────────────────
let processor;    // image preprocessing (resize 224x224, normalize pixels)
let tokenizer;    // text tokenizer (text → token IDs)
let visionModel;  // SiglipVisionModel — encodes images
let textModel;    // SiglipTextModel  — encodes text

const progressCallback = (p) => {
  if (p.status === "progress") {
    const pct = ((p.loaded / p.total) * 100).toFixed(1);
    process.stdout.write(`\r⏳ Downloading ${p.file}: ${pct}%   `);
  } else if (p.status === "done") {
    console.log(`\n✅ Loaded: ${p.file}`);
  }
};

async function loadModels() {
  console.log("🚀 Loading SigLIP2 models...");

  // AutoProcessor: resize image to 224x224 and normalize pixel values
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  // AutoTokenizer: converts text string into token IDs
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  // SiglipVisionModel: encodes images → 768-dim embedding
  // Trained with SiglipTextModel in the same shared vector space
  visionModel = await SiglipVisionModel.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  // SiglipTextModel: encodes text → 768-dim embedding
  // Same shared space as visionModel → cross-modal search works ✅
  textModel = await SiglipTextModel.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  console.log("✨ SigLIP2 ready! Vision + Text models loaded.");
}

// ─── MATH HELPERS ────────────────────────────────────────────────────────────

// L2 normalize a vector to unit length
// This is REQUIRED so cosine similarity = dot product
// And so image + text embeddings are comparable on the same scale
function l2Normalize(vec) {
  const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
  return norm === 0 ? vec : vec.map((v) => v / norm);
}

// Cosine similarity between two L2-normalized unit vectors
// Since ||a|| = ||b|| = 1, cosine_sim(a,b) = a · b
function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// Average two unit vectors and re-normalize (for hybrid AND search)
function averageEmbeddings(embA, embB) {
  const avg = embA.map((v, i) => (v + embB[i]) / 2);
  return l2Normalize(avg);
}

// ─── EMBEDDING FUNCTIONS ─────────────────────────────────────────────────────

// IMAGE EMBEDDING via SiglipVisionModel
async function getImageEmbedding(imagePath) {
  // 1. Load raw image from disk
  const image = await RawImage.read(imagePath);

  // 2. Preprocess: resize to 224x224, normalize pixels
  const inputs = await processor(image);

  // 3. Run through vision encoder
  //    pooler_output: final [CLS] pooled representation, shape [1, 768]
  const { pooler_output } = await visionModel(inputs);

  // 4. L2 normalize → unit vector
  return l2Normalize(Array.from(pooler_output.data));
}

// TEXT EMBEDDING via SiglipTextModel
async function getTextEmbedding(text) {
  // 1. Tokenize with padding + truncation for consistent tensor shape
 const inputs = await tokenizer(text, {
  padding: "max_length",
  max_length: 64,      // ← this one line fixes it
  truncation: true,
});

  // 2. Run through text encoder
  //    pooler_output: final [EOS] pooled representation, shape [1, 768]
  //    Same 768-dim space as vision model → cross-modal similarity works ✅
  const { pooler_output } = await textModel(inputs);

  // 3. L2 normalize → unit vector, directly comparable to image embeddings
  return l2Normalize(Array.from(pooler_output.data));
}

// ─── ROUTES ──────────────────────────────────────────────────────────────────

// POST /upload — Vectorize an image and store in MongoDB
app.post("/upload", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No image uploaded" });

    const embedding = await getImageEmbedding(req.file.path);

    await collection.insertOne({
      imagePath: req.file.path,
      embedding,           // normalized float[] of length 768
      createdAt: new Date(),
    });

    console.log(`📦 Stored: ${req.file.filename} (dim: ${embedding.length})`);
    res.json({ message: "Upload success", file: req.file.filename });
  } catch (err) {
    console.error("Upload error:", err);
    res.status(500).json({ error: err.message });
  }
});

// POST /search — Cross-modal semantic search
// Modes:
//   OR (Standard)  → text-only OR image-only query vector
//   AND (Hybrid)   → averaged text + image vector (must provide both)
app.post("/search", upload.single("image"), async (req, res) => {
  try {
    const { text, mode } = req.body;
    const hasText  = text && text.trim().length > 0;
    const hasImage = !!req.file;

    if (!hasText && !hasImage) {
      return res.status(400).json({ error: "Provide at least text or image" });
    }

    let queryVector;

    if (hasText && hasImage && mode === "AND") {
      // ── HYBRID: average text + image embeddings ───────────────────────────
      // Both are unit vectors in SigLIP2 shared space → safe to average
      console.log(`🔀 Hybrid (AND): averaging vectors for "${text}"`);
      const textEmb  = await getTextEmbedding(text.trim());
      const imageEmb = await getImageEmbedding(req.file.path);
      queryVector = averageEmbeddings(textEmb, imageEmb);

    } else if (hasText) {
      // ── TEXT → IMAGE search ───────────────────────────────────────────────
      console.log(`📝 Text search: "${text}"`);
      queryVector = await getTextEmbedding(text.trim());

    } else {
      // ── IMAGE → IMAGE search ──────────────────────────────────────────────
      console.log("🖼️  Image search");
      queryVector = await getImageEmbedding(req.file.path);
    }

    // Pull all stored records from MongoDB
    const allImages = await collection.find(
      {},
      { projection: { imagePath: 1, embedding: 1 } }
    ).toArray();

    if (allImages.length === 0) return res.json([]);

    // Score each stored image against the query vector
    const results = allImages.map((img) => ({
      imagePath:  img.imagePath,
      similarity: cosineSimilarity(queryVector, img.embedding),
    }));

    // Sort descending, return top 10
    results.sort((a, b) => b.similarity - a.similarity);
    const top = results.slice(0, 10);

    console.log(`🎯 Top similarity: ${(top[0]?.similarity * 100).toFixed(2)}%`);
    res.json(top);

  } catch (err) {
    console.error("Search error:", err);
    res.status(500).json({ error: err.message });
  }
});

// GET /stats — How many images are indexed
app.get("/stats", async (req, res) => {
  try {
    const count = await collection.countDocuments();
    res.json({ totalImages: count });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// POST /reindex — Re-embed all stored images (use after model change)
app.post("/reindex", async (req, res) => {
  try {
    const all = await collection.find().toArray();
    let updated = 0;
    for (const doc of all) {
      if (!fs.existsSync(doc.imagePath)) continue;
      const embedding = await getImageEmbedding(doc.imagePath);
      await collection.updateOne({ _id: doc._id }, { $set: { embedding } });
      updated++;
    }
    console.log(`🔄 Re-indexed ${updated} images`);
    res.json({ message: `Re-indexed ${updated} images` });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ─── BOOT ────────────────────────────────────────────────────────────────────
async function start() {
  await connectDB();
  await loadModels();
  const PORT = process.env.PORT || 5000;
  app.listen(PORT, () =>
    console.log(`\n🚀 Server → http://localhost:${PORT}\n`)
  );
}

start();