function About() {
    return (
      <div className="page">
        <div className="card">
          <h1>About the Model</h1>
          <p>
            This system uses a deep learning model to classify images as REAL or
            AI-generated. The image is sent to a Flask backend, processed by the
            trained model, and the prediction is returned to the frontend.
          </p>
  
          <h3>System Flow</h3>
          <p>Upload Image → Flask API → AI Model → Prediction Result</p>
        </div>
      </div>
    );
  }
  
  export default About;