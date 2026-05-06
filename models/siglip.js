import {
  AutoProcessor,
  AutoTokenizer,
  SiglipVisionModel,
  SiglipTextModel,
  RawImage,
  env,
} from "@huggingface/transformers";

import { l2Normalize } from "../utils/math.js";

env.allowLocalModels = true;
env.allowRemoteModels = true;

const MODEL_ID = "onnx-community/siglip2-base-patch16-224-ONNX";

let processor, tokenizer, visionModel, textModel;

export async function loadModels() {
  console.log("🚀 Loading SigLIP2...");

  processor = await AutoProcessor.from_pretrained(MODEL_ID);
  tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID);
  visionModel = await SiglipVisionModel.from_pretrained(MODEL_ID);
  textModel = await SiglipTextModel.from_pretrained(MODEL_ID);

  console.log("✅ Models ready");
}

export async function getImageEmbedding(path) {
  const image = await RawImage.read(path);
  const inputs = await processor(image);
  const { pooler_output } = await visionModel(inputs);
  return l2Normalize(Array.from(pooler_output.data));
}

export async function getTextEmbedding(text) {
  const inputs = await tokenizer(text, {
    padding: "max_length",
    max_length: 64,
    truncation: true,
  });

  const { pooler_output } = await textModel(inputs);
  return l2Normalize(Array.from(pooler_output.data));
}