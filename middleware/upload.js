import multer from "multer";
import path from "path";

const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    const unique = `${Date.now()}-${Math.random()}`;
    cb(null, unique + path.extname(file.originalname));
  },
});

export const upload = multer({ storage });