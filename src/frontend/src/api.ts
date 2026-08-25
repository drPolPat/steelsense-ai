import type { AllAnalysis, ReadingPoint, SensorAnalysis, SensorInfo } from './types';

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `Request failed: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function getSensors(): Promise<SensorInfo[]> {
  return request('/api/sensors');
}

export function getReadings(sensorId: string, days?: number): Promise<ReadingPoint[]> {
  const query = days ? `?days=${days}` : '';
  return request(`/api/sensors/${encodeURIComponent(sensorId)}/readings${query}`);
}

export function getSensorAnalysis(sensorId: string): Promise<SensorAnalysis> {
  return request(`/api/sensors/${encodeURIComponent(sensorId)}/analysis`);
}

export function getAllAnalysis(): Promise<AllAnalysis> {
  return request('/api/analysis');
}

export function postChat(question: string): Promise<{ answer: string }> {
  return request('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  });
}
