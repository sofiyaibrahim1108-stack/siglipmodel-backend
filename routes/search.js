import express from "express";
import { upload } from "../middleware/upload.js";
import { getCollection } from "../config/db.js";
import {
  getTextEmbedding,
  getImageEmbedding,
} from "../models/siglip.js";
import {
  cosineSimilarity,
  averageEmbeddings,
} from "../utils/math.js";

const router = express.Router();

router.post("/", upload.single("image"), async (req, res) => {
  try {
    const { text, mode } = req.body;
    const collection = getCollection();

    const hasText = text && text.trim().length > 0;
    const hasImage = !!req.file;

    let queryVector;

    // 🔀 HYBRID SEARCH
    if (hasText && hasImage && mode === "AND") {
      console.log(`🔀 Hybrid (AND): averaging vectors for "${text}"`);

      const t = await getTextEmbedding(text.trim());
      const i = await getImageEmbedding(req.file.path);

      queryVector = averageEmbeddings(t, i);

    // 📝 TEXT SEARCH
    } else if (hasText) {
      console.log(`📝 Text search: "${text}"`);

      queryVector = await getTextEmbedding(text.trim());

    // 🖼️ IMAGE SEARCH
    } else {
      console.log("🖼️  Image search");

      queryVector = await getImageEmbedding(req.file.path);
    }

    const all = await collection.find().toArray();

    const results = all.map(img => ({
      imagePath: img.imagePath,
      similarity: cosineSimilarity(queryVector, img.embedding),
    }));

    results.sort((a, b) => b.similarity - a.similarity);

    const top = results.slice(0, 10);

    // 🎯 TOP SCORE LOG
    if (top.length > 0) {
      console.log(
        `🎯 Top similarity: ${(top[0].similarity * 100).toFixed(2)}%`
      );
    }

    res.json(top);

  } catch (err) {
    console.error("❌ Search error:", err);
    res.status(500).json({ error: err.message });
  }
});

export default router;