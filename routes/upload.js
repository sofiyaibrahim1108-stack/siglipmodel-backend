import express from "express";
import { upload } from "../middleware/upload.js";
import { getCollection } from "../config/db.js";
import { getImageEmbedding } from "../models/siglip.js";

const router = express.Router();

router.post("/", upload.single("image"), async (req, res) => {
  try {
    const collection = getCollection();

    const embedding = await getImageEmbedding(req.file.path);

    await collection.insertOne({
      imagePath: req.file.path,
      embedding,
      createdAt: new Date(),
    });

    
    console.log(
      `📦 Stored: ${req.file.filename} (dim: ${embedding.length})`
    );

    res.json({ message: "Uploaded", file: req.file.filename });

  } catch (err) {
    console.error("❌ Upload error:", err);
    res.status(500).json({ error: err.message });
  }
});

export default router;