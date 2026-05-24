import express from "express";
import { upload } from "../middleware/upload.js";
import { getCollection } from "../config/db.js";
import {
  getTextEmbedding,
  getImageEmbedding
} from "../models/siglip.js";

const router = express.Router();

const DEFAULT_CONFIDENCE_THRESHOLD = 70.0;

// ─────────────────────────────────────────────────────────────
// CONFIDENCE CALCULATOR
// ─────────────────────────────────────────────────────────────

function calculateConfidence(similarity, isTextSearch) {

  let scale;
  let bias;

  if (isTextSearch) {

    scale = 55.0;
    bias = 1.5;

  } else {

    scale = 12.0;
    bias = 8.0;
  }

  const logit =
    (similarity * scale) - bias;

  const probability =
    1 / (1 + Math.exp(-logit));

  return parseFloat(
    (probability * 100).toFixed(1)
  );
}

// ─────────────────────────────────────────────────────────────
// FULL IMAGE PIPELINE
// ─────────────────────────────────────────────────────────────

function buildFullImagePipeline(queryVector) {

  return [

    {
      $addFields: {

        similarity: {

          $reduce: {

            input: {
              $range: [
                0,
                { $size: "$siglip_embedding" }
              ]
            },

            initialValue: 0,

            in: {

              $add: [

                "$$value",

                {
                  $multiply: [

                    {
                      $arrayElemAt: [
                        "$siglip_embedding",
                        "$$this"
                      ]
                    },

                    {
                      $arrayElemAt: [
                        queryVector,
                        "$$this"
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    },

    {
      $sort: {
        similarity: -1
      }
    },

    {
      $project: {

        imagePath: 1,
        similarity: 1,
        imageWidth: 1,
        imageHeight: 1,

        detections: {

          $map: {

            input: "$detections",

            as: "d",

            in: {

              class: "$$d.class",

              confidence: "$$d.confidence",

              bbox: "$$d.bbox",

              crop_image: "$$d.crop_image"
            }
          }
        },

        _id: 0
      }
    }
  ];
}

// ─────────────────────────────────────────────────────────────
// OBJECT SEARCH PIPELINE
// ─────────────────────────────────────────────────────────────

function buildObjectSearchPipeline(queryVector) {

  return [

    { $unwind: "$detections" },

    {
      $match: {
        "detections.confidence": {
          $gte: 0.45
        }
      }
    },

    {
      $addFields: {

        similarity: {

          $reduce: {

            input: {
              $range: [
                0,
                {
                  $size:
                    "$detections.siglip_embedding"
                }
              ]
            },

            initialValue: 0,

            in: {

              $add: [

                "$$value",

                {
                  $multiply: [

                    {
                      $arrayElemAt: [
                        "$detections.siglip_embedding",
                        "$$this"
                      ]
                    },

                    {
                      $arrayElemAt: [
                        queryVector,
                        "$$this"
                      ]
                    }
                  ]
                }
              ]
            }
          }
        }
      }
    },

    {
      $sort: {
        similarity: -1
      }
    },

    {
      $project: {

        imagePath: 1,

        similarity: 1,

        imageWidth: 1,

        imageHeight: 1,

        cropImage: "$detections.crop_image",

        detectionClass: "$detections.class",

        detectionConf: "$detections.confidence",

        bbox: "$detections.bbox",

        _id: 0
      }
    }
  ];
}

// ─────────────────────────────────────────────────────────────
// MAIN SEARCH ROUTE
// ─────────────────────────────────────────────────────────────

router.post(
  "/",
  upload.single("image"),
  async (req, res) => {

    try {

      const { text } = req.body;

      const collection = getCollection();

      const hasText =
        text &&
        text.trim().length > 0;

      const hasImage = !!req.file;

      if (!hasText && !hasImage) {

        return res.status(400).json({
          error:
            "Provide text or image to search"
        });
      }

      const mode = hasImage
        ? "full"
        : (
            req.body.mode === "object"
              ? "object"
              : "full"
          );

      const rawThreshold =
        parseFloat(req.body.threshold);

      const CONFIDENCE_THRESHOLD =
        (
          !isNaN(rawThreshold) &&
          rawThreshold >= 0 &&
          rawThreshold <= 100
        )
          ? rawThreshold
          : DEFAULT_CONFIDENCE_THRESHOLD;

      console.log(
        `🎯 Mode: ${mode} | Threshold: ${CONFIDENCE_THRESHOLD}%`
      );

      let scored = [];

      // ─────────────────────────────────────────────────────
      // IMAGE ONLY SEARCH
      // ─────────────────────────────────────────────────────

      if (hasImage && !hasText) {

        console.log(
          "🖼️ Image search → full image mode"
        );

        const queryVector =
          await getImageEmbedding(
            req.file.path
          );

        const raw =
          await collection
            .aggregate(
              buildFullImagePipeline(queryVector)
            )
            .toArray();

        scored = raw.map(r => ({

          type: "full",

          imagePath: r.imagePath,

          imageWidth: r.imageWidth,

          imageHeight: r.imageHeight,

          detections: r.detections || [],

          similarity: r.similarity,

          confidence:
            calculateConfidence(
              r.similarity,
              false
            )
        }));
      }

      // ─────────────────────────────────────────────────────
      // TEXT + IMAGE SEARCH
      // ─────────────────────────────────────────────────────

      else if (hasText && hasImage) {

        console.log(
          `🧠 AND Mode: "${text.trim()}" + Image`
        );

        const [
          textVector,
          imageVector
        ] = await Promise.all([

          getTextEmbedding(text.trim()),

          getImageEmbedding(req.file.path)
        ]);

        const [
          textResults,
          imageResults
        ] = await Promise.all([

          collection
            .aggregate(
              buildFullImagePipeline(textVector)
            )
            .toArray(),

          collection
            .aggregate(
              buildFullImagePipeline(imageVector)
            )
            .toArray()
        ]);

        const imageSimMap = new Map(
          imageResults.map(r => [
            r.imagePath,
            r.similarity
          ])
        );

        scored = textResults.map(r => {

          const textSim =
            Math.max(0, r.similarity);

          const imageSim =
            Math.max(
              0,
              imageSimMap.get(r.imagePath) ?? 0
            );

          const minSim =
            Math.min(textSim, imageSim);

          const avgSim =
            (textSim + imageSim) / 2;

          const combinedSim =
            (minSim * 0.75) +
            (avgSim * 0.25);

          const logit =
            (combinedSim * 36.0) - 2.2;

          const probability =
            1 / (1 + Math.exp(-logit));

          const confidence =
            parseFloat(
              (probability * 100).toFixed(1)
            );

          return {

            type: "full",

            imagePath: r.imagePath,

            imageWidth: r.imageWidth,

            imageHeight: r.imageHeight,

            detections: r.detections || [],

            similarity: combinedSim,

            confidence
          };
        });
      }

      // ─────────────────────────────────────────────────────
      // TEXT FULL SEARCH
      // ─────────────────────────────────────────────────────

      else if (
        hasText &&
        mode === "full"
      ) {

        console.log(
          `📝 Text search (full): "${text}"`
        );

        const queryVector =
          await getTextEmbedding(
            text.trim()
          );

        const raw =
          await collection
            .aggregate(
              buildFullImagePipeline(queryVector)
            )
            .toArray();

        scored = raw.map(r => ({

          type: "full",

          imagePath: r.imagePath,

          imageWidth: r.imageWidth,

          imageHeight: r.imageHeight,

          detections: r.detections || [],

          similarity: r.similarity,

          confidence:
            calculateConfidence(
              r.similarity,
              true
            )
        }));
      }

      // ─────────────────────────────────────────────────────
      // TEXT OBJECT SEARCH
      // ─────────────────────────────────────────────────────

      else if (
        hasText &&
        mode === "object"
      ) {

        console.log(
          `🔍 Text search (object): "${text}"`
        );

        const queryVector =
          await getTextEmbedding(
            text.trim()
          );

        const raw =
          await collection
            .aggregate(
              buildObjectSearchPipeline(queryVector)
            )
            .toArray();

        scored = raw.map(r => {

          const classBoost =
            text
              .toLowerCase()
              .includes(
                r.detectionClass.toLowerCase()
              )
                ? 0.25
                : 0;

          const finalSimilarity =
            r.similarity + classBoost;

          return {

            type: "crop",

            imagePath: r.imagePath,

            imageWidth: r.imageWidth,

            imageHeight: r.imageHeight,

            cropImage: r.cropImage,

            detectionClass:
              r.detectionClass,

            detectionConf:
              r.detectionConf,

            bbox: r.bbox,

            similarity: finalSimilarity,

            confidence:
              calculateConfidence(
                finalSimilarity,
                true
              )
          };
        });
      }

      // ─────────────────────────────────────────────────────
      // DEBUG
      // ─────────────────────────────────────────────────────

      console.log("🔍 All results:");

      [...scored]
        .sort(
          (a, b) =>
            b.confidence - a.confidence
        )
        .forEach(r => {

          if (r.type === "crop") {

            console.log(
              `   ${r.confidence}% [${r.detectionClass}] bbox=${JSON.stringify(r.bbox)} → ${r.cropImage}`
            );

          } else {

            console.log(
              `   ${r.confidence}% → ${r.imagePath} (${r.detections?.length || 0} boxes)`
            );
          }
        });

      // ─────────────────────────────────────────────────────
      // FINAL FILTER
      // ─────────────────────────────────────────────────────

      const results =
        scored
          .filter(
            r =>
              r.confidence >=
              CONFIDENCE_THRESHOLD
          )
          .sort(
            (a, b) =>
              b.confidence - a.confidence
          )
          .slice(0, 10);

      console.log(

        results.length > 0

          ? `✅ Found ${results.length} results above ${CONFIDENCE_THRESHOLD}%`

          : `⚠️ No results above ${CONFIDENCE_THRESHOLD}%`
      );

      return res.json({

        mode,

        total: results.length,

        results
      });

    } catch (err) {

      console.error(
        "❌ Search error:",
        err
      );

      return res.status(500).json({

        error: err.message
      });
    }
  }
);

export default router;