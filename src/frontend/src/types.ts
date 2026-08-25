export interface SensorInfo {
  sensor_id: string;
  location: string;
  description: string;
  baseline_mv: number;
}

export interface ReadingPoint {
  timestamp: string;
  reading_mv: number;
  temperature_c: number;
}

export interface DriftResult {
  detected: boolean;
  slope_mv_per_day: number;
  projected_change_mv: number;
  significance_sigma: number;
  growing_cyclic_amplitude: boolean;
  amplitude_growth_ratio: number;
  amplitude_trend_mv_per_day_per_day: number;
}

export interface SpikeEvent {
  start: string;
  end: string;
  peak_mv: number;
  peak_sigma: number;
  post_event_residual_offset_mv: number;
  lasting_offset: boolean;
}

export interface DivergenceResult {
  detected: boolean;
  peer_sensor_id: string | null;
  peer_location?: string;
  gap_slope_mv_per_day?: number;
  gap_projected_change_mv?: number;
  significance_sigma?: number;
  note?: string | null;
}

export interface SensorAnalysis {
  sensor_id: string;
  location: string;
  thermal_coefficient_mv_per_c: number;
  baseline_noise_std_mv: number;
  drift: DriftResult;
  spikes: SpikeEvent[];
  divergence: DivergenceResult;
  flags: string[];
  concern_score: number;
}

export interface RankedSensor {
  sensor_id: string;
  location: string;
  concern_score: number;
  flags: string[];
}

export interface AllAnalysis {
  sensors: Record<string, SensorAnalysis>;
  ranked_by_concern: RankedSensor[];
}

export type Severity = 'good' | 'warning' | 'serious' | 'critical';

export function severityFor(sensor: RankedSensor): Severity {
  if (sensor.flags.length === 0) return 'good';
  if (sensor.concern_score >= 8) return 'critical';
  if (sensor.concern_score >= 5) return 'serious';
  return 'warning';
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}
