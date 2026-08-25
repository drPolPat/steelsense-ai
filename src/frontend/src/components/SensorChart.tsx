import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { getReadings } from '../api';
import type { ReadingPoint, SensorAnalysis } from '../types';

// Must match BASELINE_DAYS in src/backend/data/analysis.py -- the
// commissioning window the detection pipeline calibrates against.
const BASELINE_DAYS = 14;

interface ChartRow {
  timestamp: string;
  reading_mv: number;
  peer_reading_mv?: number;
}

interface Props {
  sensorId: string;
  analysis: SensorAnalysis;
}

// Pinned to en-US regardless of the viewer's browser locale, so date labels
// stay consistent with the rest of the (English) UI.
function formatTick(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function formatFull(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

function CustomTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="tt-label">{formatFull(label)}</div>
      {payload.map((entry: any) => (
        <div key={entry.dataKey} style={{ color: entry.color }}>
          {entry.name}: {entry.value?.toFixed(2)} mV
        </div>
      ))}
    </div>
  );
}

export default function SensorChart({ sensorId, analysis }: Props) {
  const [rows, setRows] = useState<ChartRow[] | null>(null);
  const [peerLocation, setPeerLocation] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setRows(null);
    setPeerLocation(null);

    const peerId = analysis.divergence.detected ? analysis.divergence.peer_sensor_id : null;

    Promise.all([getReadings(sensorId), peerId ? getReadings(peerId) : Promise.resolve(null)])
      .then(([primary, peer]: [ReadingPoint[], ReadingPoint[] | null]) => {
        if (cancelled) return;
        const merged: ChartRow[] = primary.map((point, i) => ({
          timestamp: point.timestamp,
          reading_mv: point.reading_mv,
          peer_reading_mv: peer && peer.length === primary.length ? peer[i].reading_mv : undefined,
        }));
        setRows(merged);
        if (peer) setPeerLocation(analysis.divergence.peer_location ?? peerId);
      })
      .catch(() => {
        if (!cancelled) setRows([]);
      });

    return () => {
      cancelled = true;
    };
  }, [sensorId, analysis]);

  if (rows === null) {
    return (
      <div className="card chart-card">
        <div className="card-header">{analysis.location}</div>
        <div className="chart-card-body chart-loading">Loading readings…</div>
      </div>
    );
  }

  if (rows.length === 0) {
    return (
      <div className="card chart-card">
        <div className="card-header">{analysis.location}</div>
        <div className="chart-card-body chart-empty">No readings available.</div>
      </div>
    );
  }

  const driftStart = analysis.drift.detected
    ? rows[Math.min(Math.floor((BASELINE_DAYS / 60) * rows.length), rows.length - 1)]?.timestamp
    : null;
  const seriesEnd = rows[rows.length - 1].timestamp;

  return (
    <div className="card chart-card">
      <div className="card-header">
        {analysis.location} — reading history
        {peerLocation && ` (with ${peerLocation} for comparison)`}
      </div>
      <div className="chart-card-body">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={rows} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="var(--gridline)" strokeDasharray="0" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatTick}
              stroke="var(--baseline-axis)"
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              minTickGap={40}
            />
            <YAxis
              stroke="var(--baseline-axis)"
              tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
              width={48}
              label={{ value: 'mV', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)' }}
              domain={['auto', 'auto']}
            />
            <Tooltip content={<CustomTooltip />} />

            {driftStart && (
              <ReferenceArea
                x1={driftStart}
                x2={seriesEnd}
                fill="var(--status-warning)"
                fillOpacity={0.08}
                ifOverflow="extendDomain"
              />
            )}

            {analysis.spikes.map((spike) => (
              <ReferenceArea
                key={spike.start}
                x1={spike.start}
                x2={spike.end}
                fill={spike.lasting_offset ? 'var(--status-critical)' : 'var(--status-serious)'}
                fillOpacity={0.35}
                ifOverflow="extendDomain"
              />
            ))}

            <Line
              type="monotone"
              dataKey="reading_mv"
              name={analysis.location}
              stroke="var(--series-1)"
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
            {peerLocation && (
              <Line
                type="monotone"
                dataKey="peer_reading_mv"
                name={peerLocation}
                stroke="var(--series-2)"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            )}
            {peerLocation && <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {(analysis.spikes.length > 0 || driftStart) && (
        <div className="chart-legend">
          {analysis.spikes.length > 0 && (
            <span>
              <span className="swatch" style={{ background: 'var(--status-critical)' }} />
              spike window
            </span>
          )}
          {driftStart && (
            <span>
              <span className="swatch" style={{ background: 'var(--status-warning)' }} />
              drift period
            </span>
          )}
        </div>
      )}
    </div>
  );
}
