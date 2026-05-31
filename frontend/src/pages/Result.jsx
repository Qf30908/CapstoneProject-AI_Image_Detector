import { useLocation, Link } from "react-router-dom";

function Result() {
  const location = useLocation();
  const { result, preview } = location.state || {};

  if (!result) {
    return (
      <div className="page">
        <h2>No result found</h2>
        <Link to="/upload" className="btn">Upload Image</Link>
      </div>
    );
  }

  const label = result.class || result.label || result.prediction;
  const confidence = result.confidence;

  return (
    <div className="page">
      <div className="card result-card">
        <h1>Prediction Result</h1>

        {preview && <img src={preview} className="preview" />}

        <h2 className={label === "REAL" ? "real" : "fake"}>
          {label}
        </h2>

        <p>Confidence: {(confidence).toFixed(2)}%</p>

        <Link to="/upload" className="btn">Test Another Image</Link>
      </div>
    </div>
  );
}

export default Result;