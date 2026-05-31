import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { predictImage } from "../services/api";

function Upload() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleImage = (e) => {
    const file = e.target.files[0];
    setImage(file);
    setPreview(URL.createObjectURL(file));
  };

  const handleSubmit = async () => {
    if (!image) return alert("Please select an image");

    setLoading(true);
    const result = await predictImage(image);
    setLoading(false);

    navigate("/result", {
      state: {
        result,
        preview,
      },
    });
  };

  return (
    <div className="page">
      <div className="card">
        <h1>Upload Image</h1>
        <p>Select an image to analyze.</p>

        <input type="file" accept="image/*" onChange={handleImage} />

        {preview && <img src={preview} className="preview" />}

        <button className="btn" onClick={handleSubmit} disabled={loading}>
          {loading ? "Analyzing..." : "Analyze Image"}
        </button>
      </div>
    </div>
  );
}

export default Upload;