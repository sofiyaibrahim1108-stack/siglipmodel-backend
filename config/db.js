import { MongoClient } from "mongodb";

let client;
let collection;

export async function connectDB() {
  try {
    // 🔥 Create client ONLY after dotenv is loaded
    client = new MongoClient(process.env.MONGO_URI);

    await client.connect();

    const db = client.db(process.env.DB_NAME);
    collection = db.collection(process.env.COLLECTION_NAME);

    console.log("✅ MongoDB Connected");
  } catch (err) {
    console.error("❌ MongoDB Connection Error:", err);
    process.exit(1);
  }
}

export function getCollection() {
  if (!collection) {
    throw new Error("❌ DB not initialized. Call connectDB() first.");
  }
  return collection;
}