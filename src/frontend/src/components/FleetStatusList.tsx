import { severityFor, type RankedSensor } from '../types';

interface Props {
  sensors: RankedSensor[];
  selectedId: string | null;
  onSelect: (sensorId: string) => void;
}

export default function FleetStatusList({ sensors, selectedId, onSelect }: Props) {
  return (
    <div className="card fleet-list">
      <div className="card-header">Sensor fleet status</div>
      <div>
        {sensors.map((sensor) => (
          <button
            key={sensor.sensor_id}
            className={`fleet-row${sensor.sensor_id === selectedId ? ' selected' : ''}`}
            onClick={() => onSelect(sensor.sensor_id)}
          >
            <span className={`status-dot ${severityFor(sensor)}`} />
            <span className="location">{sensor.location}</span>
            <span className="flags">
              {sensor.flags.map((flag) => (
                <span key={flag} className="flag-chip">
                  {flag.replaceAll('_', ' ')}
                </span>
              ))}
            </span>
            <span className="score">{sensor.concern_score.toFixed(1)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
