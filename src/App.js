import React, { useState } from 'react';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as toxicity from '@tensorflow-models/toxicity';
import '@tensorflow/tfjs';

function App() {
  const [image, setImage] = useState(null);
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState('');
  const [textAnalysis, setTextAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle Image Upload and Classification
  const handleImageUpload = async (e) => {
    const imgFile = e.target.files[0];
    if (!imgFile) return;

    const imgURL = URL.createObjectURL(imgFile);
    setImage(imgURL);

    const imgElement = document.createElement('img');
    imgElement.src = imgURL;
    imgElement.onload = async () => {
      const model = await mobilenet.load();
      const predictions = await model.classify(imgElement);
      setPrediction(predictions[0]?.className || 'No result');
    };
  };

  // Handle Text Input Change
  const handleTextChange = (e) => {
    setText(e.target.value);
  };

  // Analyze Text for Toxicity
  const analyzeText = async () => {
    setLoading(true);

    const model = await toxicity.load(0.7);  // Lower threshold for better sensitivity
    const predictions = await model.classify([text]);

    setTextAnalysis(predictions);
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-6">
      <h1 className="text-4xl font-bold mb-8 text-center text-gray-800">Image Recognition and Text Analyzing with AI</h1>

      {/* Image Upload Section */}
      <div className="mb-6 w-full max-w-md">
        <h2 className="text-2xl mb-2">Image Upload</h2>
        <input
          type="file"
          onChange={handleImageUpload}
          className="mb-4 w-full p-2 border border-gray-300 rounded"
        />
        {image && <img src={image} alt="Uploaded" className="mt-4 w-full h-auto rounded-lg shadow-md" />}
        {prediction && <p className="mt-4 text-xl font-semibold">Prediction: {prediction}</p>}
      </div>

      {/* Text Input Section */}
      <div className="mb-6 w-full max-w-md">
        <h2 className="text-2xl mb-2">Text Input</h2>
        <textarea
          value={text}
          onChange={handleTextChange}
          placeholder="Enter some text..."
          className="w-full p-4 border border-gray-300 rounded h-32"
        />
        <button
          onClick={analyzeText}
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded mt-4"
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Analyze Text'}
        </button>

        {/* Sentiment Analysis Results */}
        {textAnalysis && (
          <div className="mt-4">
            <h3 className="text-lg font-semibold">Sentiment Analysis Results:</h3>
            <ul>
              {textAnalysis.map((item, index) => (
                <li key={index} className="mt-2">
                  <strong>{item.label}:</strong>{' '}
                  <span
                    className={`${
                      item.results[0].match
                        ? 'text-red-500 font-bold' // Toxic labels in red
                        : 'text-green-500' // Non-toxic labels in green
                    }`}
                  >
                    {item.results[0].match ? 'Toxic' : 'Not Toxic'}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
