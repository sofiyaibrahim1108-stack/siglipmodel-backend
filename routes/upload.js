
import express from "express";
import multer from "multer";
import fetch from "node-fetch";

const router = express.Router();


// MULTER MEMORY STORAGE


const storage = multer.memoryStorage();

const upload = multer({
  storage: storage
});


// PYTHON SERVICE URL

const PYTHON_SERVICE =
  process.env.MODEL_SERVICE_URL || "http://localhost:8000";


// UPLOAD ROUTE

router.post("/", upload.single("image"), async (req, res) => {

  try {

    // VALIDATE FILE

    if (!req.file) {
      return res.status(400).json({
        error: "No image uploaded"
      });
    }

    console.log(
      `📤 Proxying direct binary buffer to Python: ${req.file.originalname}`
    );

    // ENCODE ORIGINAL FILENAME

    const encodedFilename =
      encodeURIComponent(req.file.originalname);

    // SEND RAW BUFFER TO PYTHON
  

    const response = await fetch(
      `${PYTHON_SERVICE}/detect?filename=${encodedFilename}`,
      {
        method: "POST",

        headers: {
          "Content-Type": req.file.mimetype
        },

       body: Buffer.from(req.file.buffer)
      }
    );

    // SAFE JSON PARSE

    let data;

    try {

      data = await response.json();

    } catch {

      return res.status(500).json({
        error: "Python returned invalid JSON response"
      });
    }

    // HANDLE PYTHON ERRORS

    if (!response.ok) {

      console.error("❌ Python detect error:", data);

      return res.status(500).json({
        error: data.error || "Detection failed"
      });
    }

    // SUCCESS

    console.log(
      `✅ Detection complete — ${data.total_detections} objects stored directly by Python`
    );

    return res.status(200).json(data);

  } catch (error) {

    console.error("❌ Node upload proxy error:", error);

    return res.status(500).json({
      error: "Internal server proxy tracking failed"
    });
  }
});

export default router;