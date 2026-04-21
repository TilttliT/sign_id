import { useState } from 'react';

function Identify() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [threshold, setThreshold] = useState(0.7);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
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
    if (!file) {
      setError('Загрузите изображение подписи');
      return;
    }
    setLoading(true);
    setError(null);
    const formData = new FormData();
    formData.append('image', file);
    formData.append('threshold', threshold.toString());

    try {
      const response = await fetch('http://127.0.0.1:8000/identify', {
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
    <div className="identify-container">
      <div className="upload-box single">
        <label>📄 Подпись</label>
        <input type="file" accept="image/*" onChange={handleFileChange} />
        {preview && <img src={preview} alt="preview" className="preview" />}
      </div>

      <div className="threshold-control">
        <label>Порог уверенности: <strong>{threshold}</strong></label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.01"
          value={threshold}
          onChange={(e) => setThreshold(parseFloat(e.target.value))}
        />
        <small>Ниже порога — подпись считается неизвестной</small>
      </div>

      <button onClick={handleSubmit} disabled={loading} className="submit-btn">
        {loading ? 'Распознавание...' : 'Распознать'}
      </button>

      {error && <div className="error">❌ {error}</div>}

      {result && (
        <div className={`result-card ${!result.is_unknown ? 'success' : 'fail'}`}>
          {!result.is_unknown ? (
            <>
              <div className="result-icon">✅</div>
              <div className="result-text">
                <h3>Это подпись <strong>{result.person_name}</strong></h3>
                <p>ID: {result.person_id}</p>
                <p>Уверенность: <strong>{confidencePercent}%</strong></p>
                <p>Использованный порог: {result.applied_threshold}</p>
              </div>
            </>
          ) : (
            <>
              <div className="result-icon">❓</div>
              <div className="result-text">
                <h3>Подпись не распознана</h3>
                <p>Уверенность: {confidencePercent}% (ниже порога {threshold})</p>
                <p>{result.message || 'Такой подписи нет в базе'}</p>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default Identify;