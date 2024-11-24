import React, { useState } from "react";
import * as toxicity from "@tensorflow-models/toxicity";

const threshold = 0.9;
const model = await toxicity.load(threshold);

const ToxicityDetector: React.FC = () => {
  const [inputText, setInputText] = useState("");
  const [resultText, setResultText] = useState("Result: ");
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputText(e.target.value);
  };

  const handleDetectClick = async () => {
    setIsLoading(true);
    setResultText("Detecting...");
    
    const sentences = [inputText];
    let isToxic = false;
    let displayText = "Result: ";

    const predictions = await model.classify(sentences);

    predictions.forEach((prediction) => {
        console.log(prediction)
      if (prediction.results[0].match) {
        isToxic = true;
        displayText += `${prediction.label}, `;
      }
    });

    if (!isToxic) {
      displayText += "Not toxic";
    }

    setResultText(displayText);
    setIsLoading(false);
  };

  return (
    <div className="container">
      <h1>Toxicity Detector</h1>
      <div className="form-group">
        <input
          type="text"
          className="form-control"
          value={inputText}
          onChange={handleInputChange}
          placeholder="Enter text for detection..."
        />
      </div>
      <div className="alert alert-danger" role="alert">
        {resultText}
      </div>
      <button
        className="btn btn-warning"
        onClick={handleDetectClick}
        disabled={isLoading}
      >
        <span id="detect">{isLoading ? "Detecting..." : "Detect"}</span>
        {isLoading && <div className="loader ml-1"></div>}
      </button>
    </div>
  );
};

export default ToxicityDetector;
