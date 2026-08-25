import { useEffect, useState } from 'react';
import './App.css';
import { getAllAnalysis } from './api';
import ChatPanel from './components/ChatPanel';
import FleetStatusList from './components/FleetStatusList';
import SensorChart from './components/SensorChart';
import type { AllAnalysis } from './types';

export default function App() {
  const [analysis, setAnalysis] = useState<AllAnalysis | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    getAllAnalysis()
      .then((data) => {
        setAnalysis(data);
        setSelectedId(data.ranked_by_concern[0]?.sensor_id ?? null);
      })
      .catch((err) => setLoadError(err instanceof Error ? err.message : 'Failed to load sensor data.'));
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>SteelSense AI</h1>
        <span className="subtitle">Ironwood Crossing Bridge — Structural Health Monitoring</span>
        <span className="synthetic-badge">Synthetic demo data</span>
      </header>

      <div className="app-body">
        <ChatPanel />

        <div className="dashboard-column">
          {loadError && <div className="card chart-empty">{loadError}</div>}
          {analysis && (
            <FleetStatusList
              sensors={analysis.ranked_by_concern}
              selectedId={selectedId}
              onSelect={setSelectedId}
            />
          )}
          {analysis && selectedId && (
            <SensorChart sensorId={selectedId} analysis={analysis.sensors[selectedId]} />
          )}
        </div>
      </div>
    </div>
  );
}
