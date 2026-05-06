import express from "express";
import { getCollection } from "../config/db.js";

const router = express.Router();

// GET /stats → total number of indexed images
router.get("/", async (req, res) => {
  try {
    const collection = getCollection();

    const totalImages = await collection.countDocuments();

    res.json({ totalImages });
  } catch (err) {
    console.error("Stats error:", err);
    res.status(500).json({ error: err.message });
  }
});

export default router;