import { useState } from 'react';
import Verify from './components/Verify';
import Identify from './components/Identify';
import './styles/App.css';

function App() {
  const [activeTab, setActiveTab] = useState('verify');

  return (
    <div className="app">
      <h1>️ Распознавание подписей</h1>
      <div className="tabs">
        <button
          className={activeTab === 'verify' ? 'active' : ''}
          onClick={() => setActiveTab('verify')}
        >
           Сравнить две подписи
        </button>

      </div>
      <div className="tab-content">
        {activeTab === 'verify' && <Verify />}
      </div>
    </div>
  );
}

export default App;