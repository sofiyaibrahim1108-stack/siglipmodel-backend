import express from "express";
import { getCollection } from "../config/db.js";

const router = express.Router();

router.get("/", async (req, res) => {
  try {
    const total = await getCollection().countDocuments();
    res.json({ totalImages: total });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

export default router;