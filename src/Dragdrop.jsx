import React, { useState } from "react";
import { useNavigate } from "react-router-dom"; // For navigation
import line from "../public/lines.png";
import "./Dragdrop.css";
import Bufferinganimation from "./Bufferinganimation"; // Import animation page

export const Dragdrop = () => {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false); // Track animation state
  const navigate = useNavigate();

  // Handle file selection from input
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // Handle drag-and-drop file
  const handleDrop = (e) => {
    e.preventDefault();
    setFile(e.dataTransfer.files[0]);
  };

  // Prevent default on drag over
  const handleDragOver = (e) => {
    e.preventDefault();
  };

  // Handle "Detect Now" button click
  const handleDetect = async () => {
    if (!file) {
      alert("Please upload a video file first.");
      return;
    }
  
    setIsAnalyzing(true); // Show buffering animation
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const response = await fetch("http://127.0.0.1:8000/predict", { // Replace with your API URL
        method: "POST",
        body: formData,
        headers: {
          "Accept": "application/json"
        }
      });
  
      if (!response.ok) {
        throw new Error("Failed to analyze the video.");
      }
  
      const data = await response.json();
      console.log("Prediction result:", data);
  
      // Navigate to results page with prediction data
      navigate("/results", { state: { prediction: data.prediction, confidence: data.confidence, videoUrl: URL.createObjectURL(file)} });
  
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while processing the video.");
    } finally {
      setIsAnalyzing(false);
    }
  };
  

  // Show buffering animation if analyzing
  if (isAnalyzing) {
    return <Bufferinganimation />;
  }

  return (
    <div className="work-container">
      <div className="step">
        <div className="step-line">
          <img src={line} width="100%" height="100%" alt="line" />
        </div>
        <div className="steplist-cont">
          <div className="step-list">
            <span>Go to files and </span>
            <span>upload your video</span>
          </div>
          <div className="step-list">
            <span>Analyzing </span>
            <span>your videos</span>
          </div>
          <div className="step-list">
            <span>Get your result</span>
          </div>
        </div>
        <div className="step-btn" style={{ position: "absolute", top: -42, right: -20, backgroundColor: "#000914" }}>1</div>
        <div className="step-btn" style={{ position: "absolute", top: "48%", left: -19, backgroundColor: "#000914" }}>2</div>
        <div className="step-btn" style={{ position: "absolute", bottom: -34, right: -20, backgroundColor: "#000914" }}>3</div>
      </div>
      <div className="upload-container">
        <div className="drop-zone" onDrop={handleDrop} onDragOver={handleDragOver}>
          {file ? file.name : "📎 Drag or Drop files"}
          <input type="file" accept="video/*" onChange={handleFileChange} className="hidden-input" id="fileUpload" />
        </div>
        <label htmlFor="fileUpload" className="upload-label">
          Or click to upload
        </label>
        <button onClick={handleDetect} className="detect-button">
          Detect now
        </button>
      </div>
    </div>
  );
};
