import express from "express";
import { upload } from "../middleware/upload.js";
import { getCollection } from "../config/db.js";
import { getTextEmbedding, getImageEmbedding } from "../models/siglip.js";

const router = express.Router();

const DEFAULT_CONFIDENCE_THRESHOLD = 0.0; // Fallback if frontend sends nothing (frontend always sends threshold)

/**
 * Maps raw cosine similarity scores to human-readable confidence scores (0-100%)
 * using separate calibration parameters for text queries vs. image queries.
 */
function calculateConfidence(similarity, isTextSearch) {
  let scale, bias;

  if (isTextSearch) {
    // 🔧 TEXT TUNING: Raw score 0.06 range-la vandhale 80%+ mela pokura madhiri scale panni iruken
    scale = 55.0;
    bias  = 1.5;
  } else {
    // 🖼️ IMAGE TUNING: Image search calculation accurate-ah iruka intha scale
    scale = 12.0;
    bias  = 8.0;
  }

  const logit = (similarity * scale) - bias;
  const probability = 1 / (1 + Math.exp(-logit)); // Sigmoid function

  return parseFloat((probability * 100).toFixed(1));
}

/**
 * Builds a MongoDB aggregation pipeline that computes dot product similarity
 * between stored embeddings and a given query vector.
 * Since embeddings are L2-normalized, dot product == cosine similarity.
 */
function buildSimilarityPipeline(queryVector) {
  return [
    {
      $addFields: {
        similarity: {
          $reduce: {
            input: { $range: [0, { $size: "$embedding" }] },
            initialValue: 0,
            in: {
              $add: [
                "$$value",
                {
                  $multiply: [
                    { $arrayElemAt: ["$embedding", "$$this"] },
                    { $arrayElemAt: [queryVector, "$$this"] }
                  ]
                }
              ]
            }
          }
        }
      }
    },
    { $sort: { similarity: -1 } },
    { $project: { imagePath: 1, similarity: 1, _id: 0 } }
  ];
}

router.post("/", upload.single("image"), async (req, res) => {
  try {
    const { text } = req.body;
    const collection = getCollection();

    const hasText  = text && text.trim().length > 0;
    const hasImage = !!req.file;

    if (!hasText && !hasImage) {
      return res.status(400).json({ error: "Provide text or image to search" });
    }

    // ✅ Read threshold from frontend — fallback to default if missing or invalid
    const rawThreshold = parseFloat(req.body.threshold);
    const CONFIDENCE_THRESHOLD = (!isNaN(rawThreshold) && rawThreshold >= 0 && rawThreshold <= 100)
      ? rawThreshold
      : DEFAULT_CONFIDENCE_THRESHOLD;

    console.log(`🎯 Using confidence threshold: ${CONFIDENCE_THRESHOLD}%`);

    let scored = [];

    // 🚀 CASE 1: MULTIMODAL "AND" OPTION (Both text and image provided)
    if (hasText && hasImage) {
      console.log(`🧠 AND Mode Active: Text ("${text.trim()}") + Image reference`);

      const [textVector, imageVector] = await Promise.all([
        getTextEmbedding(text.trim()),
        getImageEmbedding(req.file.path)
      ]);

      // Fetch similarity for both vectors from DB
      const [textResults, imageResults] = await Promise.all([
        collection.aggregate(buildSimilarityPipeline(textVector)).toArray(),
        collection.aggregate(buildSimilarityPipeline(imageVector)).toArray()
      ]);

      // Map imageResults by imagePath for quick lookup
      const imageSimMap = new Map(imageResults.map(r => [r.imagePath, r.similarity]));

      scored = textResults.map(r => {
        const textSim  = Math.max(0, r.similarity);
        const imageSim = Math.max(0, imageSimMap.get(r.imagePath) ?? 0);

        // Use an intersection blend (75% on the lower match + 25% on average match)
        // to strictly filter out wrong assets during dual query matches.
        const minSim = Math.min(textSim, imageSim);
        const avgSim = (textSim + imageSim) / 2;
        const combinedSim = (minSim * 0.75) + (avgSim * 0.25);

        // Strict SigLIP Multi-modal logistic scaling
        const scale = 36.0;
        const bias  = 2.2;

        const logit = (combinedSim * scale) - bias;
        const probability = 1 / (1 + Math.exp(-logit));
        const confidence = parseFloat((probability * 100).toFixed(1));

        return {
          imagePath:  r.imagePath,
          similarity: combinedSim,
          confidence: confidence
        };
      });

    // 📝 CASE 2: TEXT ONLY search
    } else if (hasText) {
      console.log(`📝 Text search: "${text}"`);
      const queryVector = await getTextEmbedding(text.trim());

      const raw = await collection.aggregate(buildSimilarityPipeline(queryVector)).toArray();

      scored = raw.map(r => ({
        imagePath:  r.imagePath,
        similarity: r.similarity,
        confidence: calculateConfidence(r.similarity, true)
      }));

    // 🖼️ CASE 3: IMAGE ONLY search
    } else {
      console.log("🖼️  Image search");
      const queryVector = await getImageEmbedding(req.file.path);

      const raw = await collection.aggregate(buildSimilarityPipeline(queryVector)).toArray();

      scored = raw.map(r => ({
        imagePath:  r.imagePath,
        similarity: r.similarity,
        confidence: calculateConfidence(r.similarity, false)
      }));
    }

    // Debug log
    console.log("🔍 All item confidence scores:");
    [...scored]
      .sort((a, b) => b.confidence - a.confidence)
      .forEach(r => console.log(`   ${r.confidence}% (raw: ${r.similarity.toFixed(3)}) → ${r.imagePath}`));

    // ✅ Filter by dynamic threshold from frontend, sort, return top 10
    const results = scored
      .filter(r => r.confidence >= CONFIDENCE_THRESHOLD)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 10);

    if (results.length > 0) {
      console.log(`✅ Found ${results.length} results above ${CONFIDENCE_THRESHOLD}% confidence`);
    } else {
      console.log(`⚠️  No results above ${CONFIDENCE_THRESHOLD}% confidence`);
    }

    res.json(results);

  } catch (err) {
    console.error("❌ Search error:", err);
    res.status(500).json({ error: err.message });
  }
});

export default router;