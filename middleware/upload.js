import multer from "multer";
import path from "path";

const storage = multer.diskStorage({
  destination: "uploads/",

  filename: (req, file, cb) => {
    const unique = `${Date.now()}-${Math.random()}`;

    cb(null, unique + path.extname(file.originalname));
  },
});

// ✅ Allow only image files
const fileFilter = (req, file, cb) => {
  const allowedTypes = [
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/webp",
  ];

  if (!allowedTypes.includes(file.mimetype)) {
    return cb(
      new Error("Only PNG, JPG, JPEG, and WEBP images are allowed")
    );
  }

  cb(null, true);
};

export const upload = multer({
  storage,

  // ✅ File validation
  fileFilter,

  // ✅ File size limit (5MB)
  limits: {
    fileSize: 5 * 1024 * 1024,
  },
});