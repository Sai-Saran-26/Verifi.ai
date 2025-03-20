import React from "react";
import "./Results.css"; 
import { FaCheck } from "react-icons/fa"; // If using react-icons
import placeholder from "../public/placeholder.png";

export const Results = () => {
  return (
    <div className="results-container">
      <div className="results-card">
        {/* Blurred Image Placeholder */}
        <div className="image-container">
          <img
            src={placeholder} // Replace with your actual image
            alt="Analyzed"
            className="blurred-image"
          />
          <div className="detected-text">
          <FaCheck className="check-icon" size={24} />
            {/* <Check className="check-icon" size={24} /> */}
            <span>Detected</span>
          </div>
        </div>

        {/* Buttons Section */}
        <div className="button-container">
          <button className="original-button">Original</button>
          <div className="confidence-score">Confidence score : 95</div>
        </div>
      </div>
    </div>
  );
};
