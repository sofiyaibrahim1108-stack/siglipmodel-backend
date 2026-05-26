// 🔥 VERY IMPORTANT: load env FIRST
import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";
import fs from "fs";

import { connectDB } from "./config/db.js";
import { loadModels } from "./models/siglip.js";

import uploadRoutes from "./routes/upload.js";
import searchRoutes from "./routes/search.js";
import statsRoutes from "./routes/stats.js";

const app = express();

//  MIDDLEWARE
app.use(cors());
app.use(express.json());

//  STATIC FOLDER 
if (!fs.existsSync("uploads")) {
  fs.mkdirSync("uploads");
}
app.use("/uploads", express.static("uploads"));

//  ROUTES 
app.use("/upload", uploadRoutes);
app.use("/search", searchRoutes);
app.use("/stats", statsRoutes);

//  HEALTH CHECK 
app.get("/", (req, res) => {
  res.send("🚀 SigLIP2 API is running");
});

//  START SERVER
async function start() {
  try {
    console.log("⏳ Starting server...");

    
    await connectDB();     
    await loadModels();    

    const PORT = process.env.PORT || 5000;

    app.listen(PORT, () => {
      console.log(`\n🚀 Server running on http://localhost:${PORT}\n`);
    });

  } catch (err) {
    console.error("❌ Startup error:", err);
    process.exit(1);
  }
}

start();