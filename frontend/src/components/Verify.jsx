import { useState } from 'react';

function Verify() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [preview1, setPreview1] = useState(null);
  const [preview2, setPreview2] = useState(null);
  const [threshold, setThreshold] = useState(0.65);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e, setFile, setPreview) => {
    const file = e.target.files[0];
    if (file) {
      setFile(file);
      const url = URL.createObjectURL(file);
      setPreview(url);
      setResult(null);
      setError(null);
    }
  };

  const handleSubmit = async () => {
    if (!file1 || !file2) {
      setError('Загрузите оба изображения');
      return;
    }
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image1', file1);
    formData.append('image2', file2);
    formData.append('threshold', threshold.toString());

    try {
      const response = await fetch('http://127.0.0.1:8000/verify', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Ошибка сервера');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : 0;

  return (
    <div className="verify-container">
      <div className="upload-area">
        <div className="upload-box">
          <label>📄 Подпись 1</label>
          <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setFile1, setPreview1)} />
          {preview1 && <img src={preview1} alt="preview" className="preview" />}
        </div>
        <div className="upload-box">
          <label>📄 Подпись 2</label>
          <input type="file" accept="image/*" onChange={(e) => handleFileChange(e, setFile2, setPreview2)} />
          {preview2 && <img src={preview2} alt="preview" className="preview" />}
        </div>
      </div>

      <div className="threshold-control">
        <label>Порог схожести: <strong>{threshold}</strong></label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
        />
        <small>Чем выше порог, тем строже сравнение</small>
      </div>

      <button onClick={handleSubmit} disabled={loading} className="submit-btn">
        {loading ? 'Сравнение...' : 'Сравнить'}
      </button>

      {error && <div className="error">❌ {error}</div>}

      {result && (
        <div className={`result-card ${result.match ? 'success' : 'fail'}`}>
          {result.match ? (
            <>
              <div className="result-icon">✅</div>
              <div className="result-text">
                <h3>Подписи принадлежат ОДНОМУ человеку</h3>
                <p>Уверенность: <strong>{confidencePercent}%</strong></p>
                <p>Использованный порог: {result.applied_threshold}</p>
              </div>
            </>
          ) : (
            <>
              <div className="result-icon">❌</div>
              <div className="result-text">
                <h3>Подписи принадлежат РАЗНЫМ людям</h3>
                <p>Уверенность: <strong>{confidencePercent}%</strong></p>
                <p>Использованный порог: {result.applied_threshold}</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default Verify;