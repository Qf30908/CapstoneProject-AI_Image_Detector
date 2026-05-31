import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { predictImage } from "../services/api";

function Upload() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const resizeImageTo32px = (file) => {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const reader = new FileReader();

      reader.onload = (e) => {
        img.src = e.target.result;
      };

      img.onload = () => {
        const newWidth = 32;
        const newHeight = Math.round((img.height * newWidth) / img.width);

        const canvas = document.createElement("canvas");
        canvas.width = newWidth;
        canvas.height = newHeight;

        const ctx = canvas.getContext("2d");

        // Browser bilinear-like smoothing
        ctx.imageSmoothingEnabled = true;
        ctx.imageSmoothingQuality = "medium";

        ctx.drawImage(img, 0, 0, newWidth, newHeight);

        canvas.toBlob(
          (blob) => {
            if (!blob) {
              reject("Resize failed");
              return;
            }

            const resizedFile = new File([blob], "resized_32px.png", {
              type: "image/png",
            });

            resolve(resizedFile);
          },
          "image/png"
        );
      };

      img.onerror = () => reject("Image load failed");
      reader.onerror = () => reject("File read failed");

      reader.readAsDataURL(file);
    });
  };

  const handleImage = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleSubmit = async () => {
    if (!image) return alert("Please select an image");

    try {
      setLoading(true);

      const resizedImage = await resizeImageTo32px(image);
      const resizedPreview = URL.createObjectURL(resizedImage);

      const result = await predictImage(resizedImage);

      navigate("/result", {
        state: {
          result,
          preview: resizedPreview,
        },
      });
    } catch (error) {
      console.error(error);
      alert("Image processing failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1>Upload Image</h1>
        <p>Select an image to analyze.</p>

        <input type="file" accept="image/*" onChange={handleImage} />

        {preview && <img src={preview} className="preview" alt="Preview" />}

        <button className="btn" onClick={handleSubmit} disabled={loading}>
          {loading ? "Resizing & Analyzing..." : "Analyze Image"}
        </button>
      </div>
    </div>
  );
}

export default Upload;
