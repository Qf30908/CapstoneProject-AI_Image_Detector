import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="page hero">
      <h1>AI-Based Image Authenticity Detector</h1>
      <p>Upload an image and detect whether it is REAL or AI-generated.</p>
      <Link className="btn" to="/upload">Start Detection</Link>
    </div>
  );
}

export default Home;