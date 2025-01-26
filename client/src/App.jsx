import React, { useState } from 'react';

function App() {
  const [image, setImage] = useState(null);
  const [processedImage, setProcessedImage] = useState(null);

  const handleImageUpload = (event) => {
    setImage(event.target.files[0]);
  };

  const detectObjects = async () => {
    if (!image) {
      alert('Please upload an image first.');
      return;
    }

    const formData = new FormData();
    formData.append('image', image);

    try {
      const response = await fetch('http://localhost:5000/detect-objects', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Failed to detect objects');
      }

      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setProcessedImage(imageUrl);
    } catch (error) {
      console.error('Error:', error);
      alert('Failed to detect objects');
    }
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>Object Detection App</h1>
      <input type="file" accept="image/*" onChange={handleImageUpload} />
      <br /><br />

      <button onClick={detectObjects}>Detect Objects</button>

      {processedImage && (
        <div>
          <h2>Processed Image</h2>
          <img src={processedImage} alt="Processed" style={{ maxWidth: '100%', height: 'auto' }} />
        </div>
      )}
    </div>
  );
}

export default App;