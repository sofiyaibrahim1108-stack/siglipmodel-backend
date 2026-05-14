import express from "express";
import { upload } from "../middleware/upload.js";
import { getCollection } from "../config/db.js";
import { getTextEmbedding, getImageEmbedding } from "../models/siglip.js";
import { cosineSimilarity } from "../utils/math.js";

const router = express.Router();

// Thresholds — text→image scores are naturally lower than image→image
const TEXT_SIMILARITY_THRESHOLD  = 0.05; // 15% for text → image
const IMAGE_SIMILARITY_THRESHOLD = 0.80; // 80% for image → image

router.post("/", upload.single("image"), async (req, res) => {
  try {
    const { text } = req.body;
    const collection = getCollection();

    const hasText  = text && text.trim().length > 0;
    const hasImage = !!req.file;

    if (!hasText && !hasImage) {
      return res.status(400).json({ error: "Provide text or image to search" });
    }

    let queryVector;
    let threshold;

    // 📝 TEXT → IMAGE search
    if (hasText) {
      console.log(`📝 Text search: "${text}"`);
      queryVector = await getTextEmbedding(text.trim());
      threshold   = TEXT_SIMILARITY_THRESHOLD;

    // 🖼️ IMAGE → IMAGE search
    } else {
      console.log("🖼️  Image search");
      queryVector = await getImageEmbedding(req.file.path);
      threshold   = IMAGE_SIMILARITY_THRESHOLD;
    }

    const all = await collection.find().toArray();

    // Score all images
    const scored = all.map(img => ({
      imagePath:  img.imagePath,
      similarity: cosineSimilarity(queryVector, img.embedding),
    }));

    // Debug log — shows actual scores so threshold can be tuned
    console.log("🔍 All similarity scores:");
    [...scored]
      .sort((a, b) => b.similarity - a.similarity)
      .forEach(r => console.log(`   ${(r.similarity * 100).toFixed(2)}% → ${r.imagePath}`));

    // Filter, sort, return top 10
    const results = scored
      .filter(r => r.similarity >= threshold)
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 10);

    if (results.length > 0) {
      console.log(`🎯 Found ${results.length} results above ${threshold * 100}%`);
    } else {
      console.log(`⚠️  No results above ${threshold * 100}%`);
    }

    res.json(results);

  } catch (err) {
    console.error("❌ Search error:", err);
    res.status(500).json({ error: err.message });
  }
});

export default router;