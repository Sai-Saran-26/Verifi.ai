import React from "react";
import "./Bufferinganimation.css";

const Bufferinganimation = () => {
  return (
    <div className="buffering-container">
      <div className="buffering-box">
        <div className="loading-spinner"></div>
        <p className="loading-text">Analyzing</p>
      </div>
      <button className="detecting-button">Detecting</button>
    </div>
  );
};

export default Bufferinganimation;
