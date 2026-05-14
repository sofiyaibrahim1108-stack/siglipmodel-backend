import {
  AutoProcessor,
  AutoTokenizer,
  SiglipVisionModel,
  SiglipTextModel,
  RawImage,
  env,
} from "@huggingface/transformers";

import { l2Normalize } from "../utils/math.js";

// Use local cache first, only download if missing
env.allowLocalModels = true;
env.allowRemoteModels = true;

const MODEL_ID = "onnx-community/siglip2-large-patch16-384-ONNX";

// Different thresholds for each search type:
// Text→Image scores are naturally lower (25-55%) even for perfect matches
// Image→Image scores are high (85-100%) because same modality
 export const TEXT_SIMILARITY_THRESHOLD = 0.05 // 20% for text search
export const IMAGE_SIMILARITY_THRESHOLD = 0.80; // 80% for image search

// SigLIP2 text encoder max token length — must be explicit to avoid RangeError
const TEXT_MAX_LENGTH = 64;

let processor, tokenizer, visionModel, textModel;

const progressCallback = (p) => {
  if (p.status === "progress") {
    const pct = ((p.loaded / p.total) * 100).toFixed(1);
    process.stdout.write(`\r⏳ Downloading ${p.file}: ${pct}%   `);
  } else if (p.status === "done") {
    console.log(`\n✅ Loaded: ${p.file}`);
  }
};

export async function loadModels() {
  console.log("🚀 Loading SigLIP2 so400m-patch16-naflex...");

  // naflex processor handles variable resolution images automatically
  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  // Vision encoder → 1152-dim image embeddings
  visionModel = await SiglipVisionModel.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  // Text encoder → 1152-dim text embeddings
  // Trained in the SAME shared vector space as visionModel ✅
  textModel = await SiglipTextModel.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  console.log("✨ SigLIP2 so400m ready!");
}

// IMAGE EMBEDDING via SiglipVisionModel
// Returns: L2-normalized float[] of length 1152
export async function getImageEmbedding(imagePath) {
  const image = await RawImage.read(imagePath);
  const inputs = await processor(image);
  const { pooler_output } = await visionModel(inputs);
  return l2Normalize(Array.from(pooler_output.data));
}

// TEXT EMBEDDING via SiglipTextModel
// Returns: L2-normalized float[] of length 1152 (same space as image ✅)
export async function getTextEmbedding(text) {
  const inputs = await tokenizer(text, {
    padding: "max_length",
    max_length: TEXT_MAX_LENGTH,
    truncation: true,
  });
  const { pooler_output } = await textModel(inputs);
  return l2Normalize(Array.from(pooler_output.data));
}