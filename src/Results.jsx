import React, { useState, useRef, useEffect } from "react";
import { useLocation } from "react-router-dom";
import "./Results.css";
import { FaCheck } from "react-icons/fa";

export const Results = () => {
  const location = useLocation();
  const { prediction, confidence, videoUrl } = location.state || {};
  const videoRef = useRef(null);
  const [videoDimensions, setVideoDimensions] = useState({ width: 0, height: 270 }); // Default height

  useEffect(() => {
    const updateSize = () => {
      if (videoRef.current) {
        setVideoDimensions({
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight
        });
      }
    };

    if (videoRef.current) {
      videoRef.current.addEventListener("loadedmetadata", updateSize);
    }

    return () => {
      if (videoRef.current) {
        videoRef.current.removeEventListener("loadedmetadata", updateSize);
      }
    };
  }, [videoUrl]);

  return (
    <div className="results-container">
      <div className="results-card">
        {/* Video Container */}
        <div className="image-container">
          {videoUrl ? (
            <video
              ref={videoRef}
              controls
              className="analyzed-video"
              style={{
                maxHeight: "100%", // Maintain original height while fitting
                maxWidth: "100%", 
                objectFit: "contain" // Prevent distortion
              }}
            >
              <source src={videoUrl} type="video/mp4" />
              Your browser does not support the video tag.
            </video>
          ) : (
            <p>No video available</p>
          )}
        </div>

        {/* Detection Status */}
        <div className="detected-text">
          <FaCheck className="check-icon" size={24} />
          <span>{prediction ? "Detected" : "Not Detected"}</span>
        </div>

        {/* Buttons Section */}
        <div className="button-container">
          <button className="original-button">{prediction}</button>
          <div className="confidence-score">
            Confidence score: {confidence ? (1 - confidence) * 100 : "N/A"}
          </div>
        </div>
      </div>
    </div>
  );
};
