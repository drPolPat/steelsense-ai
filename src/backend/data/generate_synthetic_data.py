"""Synthetic sensor data generator for SteelSense AI.

Generates a fully synthetic, illustrative dataset simulating Hall-effect
magnetic stress sensors embedded in a fictional steel bridge. NONE of this
data is real; it is produced by simple parametric models chosen to look
plausible for a portfolio demo, not to reflect any real structure.

Physical framing (for realism, not measured fact):
    Hall-effect sensors on ferrous structural steel report a voltage that
    tracks local magnetic permeability, which in turn shifts with internal
    mechanical stress (the magnetoelastic effect). Readings are also
    cross-sensitive to temperature, which is why ambient temperature is
    simulated and included alongside each reading.

Running this script (re)writes the sample dataset under data/sample/:
    - sensor_readings.csv     raw per-sensor time series
    - sensors.json            structure + sensor metadata
    - scenario_ground_truth.json  the anomaly parameters used to generate
                               the data, for later eval reference. This is
                               NOT something the agent sees at inference
                               time -- it exists so we can grade the
                               agent's answers against a known answer key.

Usage:
    python -m src.backend.data.generate_synthetic_data
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

STRUCTURE = {
    "structure_id": "ironwood-crossing",
    "name": "Ironwood Crossing Bridge",
    "type": "steel truss highway bridge",
    "location": "Fictional, illustrative location -- not a real structure",
    "sensor_type": "embedded single-axis Hall-effect magnetic stress sensor",
}

# Time range for the generated dataset: 60 days of hourly readings.
START = pd.Timestamp("2026-05-01 00:00:00")
END = pd.Timestamp("2026-06-30 00:00:00")
SAMPLE_INTERVAL = "1h"

# Reference temperature the sensors are nominally calibrated at.
REFERENCE_TEMP_C = 15.0


@dataclass
class SensorProfile:
    sensor_id: str
    location: str
    description: str
    baseline_mv: float
    noise_std_mv: float
    thermal_coeff_mv_per_c: float
    # Optional anomaly injector: fn(t_hours, t_days) -> mV offset array
    anomaly: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    anomaly_note: str = ""
    ground_truth: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Anomaly signal models
# ---------------------------------------------------------------------------
# Each function takes t_hours / t_days (arrays covering the full series,
# starting at 0) and returns an mV offset to add on top of the sensor's
# normal baseline + noise + thermal response.

def no_anomaly(t_hours: np.ndarray, t_days: np.ndarray) -> np.ndarray:
    return np.zeros_like(t_hours)


def fatigue_cyclic_drift(
    t_hours: np.ndarray,
    t_days: np.ndarray,
    *,
    cycle_period_hours: float = 6.0,
    amplitude_start_mv: float = 0.15,
    amplitude_growth_per_day_mv: float = 0.012,
    mean_drift_per_day_mv: float = 0.03,
) -> np.ndarray:
    """Gradual fatigue signature: cyclic (traffic-load) oscillation whose
    amplitude grows over time, superimposed on a slowly rising mean offset
    (progressive permanent-set drift). Represents cumulative fatigue under
    repeated cyclic loading rather than a single acute event.
    """
    growing_amplitude = amplitude_start_mv + amplitude_growth_per_day_mv * t_days
    oscillation = growing_amplitude * np.sin(2 * np.pi * t_hours / cycle_period_hours)
    mean_drift = mean_drift_per_day_mv * t_days
    return oscillation + mean_drift


def acute_stress_spike(
    t_hours: np.ndarray,
    t_days: np.ndarray,
    *,
    spike_start_day: float,
    spike_duration_hours: float = 3.0,
    spike_magnitude_mv: float = 4.5,
    residual_fraction: float = 0.18,
    settle_days: float = 6.0,
) -> np.ndarray:
    """Sudden spike event (e.g. an overload or impact event) followed by a
    partial, gradual settle to a slightly elevated permanent baseline --
    representing minor residual/permanent deformation.
    """
    offset = np.zeros_like(t_hours)
    spike_start_hour = spike_start_day * 24
    spike_end_hour = spike_start_hour + spike_duration_hours

    during_spike = (t_hours >= spike_start_hour) & (t_hours < spike_end_hour)
    ramp = (t_hours[during_spike] - spike_start_hour) / spike_duration_hours
    # Fast rise, slightly slower fall within the spike window itself.
    offset[during_spike] = spike_magnitude_mv * np.sin(np.pi * ramp)

    after_spike = t_hours >= spike_end_hour
    days_after = (t_hours[after_spike] - spike_end_hour) / 24
    residual_target = spike_magnitude_mv * residual_fraction
    settle = residual_target * (1 - np.exp(-days_after / settle_days))
    offset[after_spike] = settle

    return offset


def localized_divergence(
    t_hours: np.ndarray,
    t_days: np.ndarray,
    *,
    divergence_start_day: float,
    rate_per_day_mv: float = 0.09,
) -> np.ndarray:
    """Slow-onset divergence from an otherwise-paired/correlated sensor,
    representing a localized structural issue affecting only this sensor's
    section (e.g. a developing crack or connection loosening nearby).
    """
    offset = np.zeros_like(t_hours)
    after_start = t_days >= divergence_start_day
    offset[after_start] = rate_per_day_mv * (t_days[after_start] - divergence_start_day)
    return offset


# ---------------------------------------------------------------------------
# Sensor layout
# ---------------------------------------------------------------------------
# 8 sensor locations on the fictional bridge. Most behave normally; three
# carry an injected anomaly covering each category called out in the
# project brief: gradual fatigue drift, an acute stress spike, and
# cross-sensor divergence between a normally-correlated pair.

SENSORS: list[SensorProfile] = [
    SensorProfile(
        sensor_id="beam-1a",
        location="Beam 1A",
        description="Main span beam, north side, near-shore end",
        baseline_mv=120.0,
        noise_std_mv=0.35,
        thermal_coeff_mv_per_c=0.18,
    ),
    SensorProfile(
        sensor_id="beam-1b",
        location="Beam 1B",
        description="Main span beam, south side, near-shore end (paired with Beam 1A)",
        baseline_mv=119.6,
        noise_std_mv=0.35,
        thermal_coeff_mv_per_c=0.18,
        anomaly=lambda h, d: localized_divergence(h, d, divergence_start_day=40),
        anomaly_note=(
            "Localized divergence starting day 40: slowly pulls away from its "
            "normally-correlated pair, Beam 1A, suggesting a developing "
            "localized issue rather than a structure-wide effect."
        ),
        ground_truth={
            "category": "cross_sensor_divergence",
            "paired_with": "beam-1a",
            "start_day": 40,
            "rate_per_day_mv": 0.09,
        },
    ),
    SensorProfile(
        sensor_id="beam-2a",
        location="Beam 2A",
        description="Main span beam, north side, mid-span",
        baseline_mv=121.5,
        noise_std_mv=0.4,
        thermal_coeff_mv_per_c=0.20,
    ),
    SensorProfile(
        sensor_id="beam-2b",
        location="Beam 2B",
        description="Main span beam, south side, mid-span",
        baseline_mv=121.1,
        noise_std_mv=0.4,
        thermal_coeff_mv_per_c=0.20,
        anomaly=lambda h, d: acute_stress_spike(h, d, spike_start_day=35),
        anomaly_note=(
            "Acute stress spike around day 35 (~3 hour event), consistent with "
            "a sudden overload or impact, followed by a small residual offset "
            "indicating minor permanent deformation."
        ),
        ground_truth={
            "category": "acute_stress_spike",
            "spike_start_day": 35,
            "spike_duration_hours": 3.0,
            "spike_magnitude_mv": 4.5,
            "residual_fraction": 0.18,
        },
    ),
    SensorProfile(
        sensor_id="beam-3a",
        location="Beam 3A",
        description="Main span beam, north side, far-shore end",
        baseline_mv=118.8,
        noise_std_mv=0.35,
        thermal_coeff_mv_per_c=0.19,
        anomaly=lambda h, d: fatigue_cyclic_drift(h, d),
        anomaly_note=(
            "Gradual fatigue signature: cyclic oscillation amplitude grows "
            "over the full 60-day window alongside a slowly rising mean "
            "offset, consistent with cumulative fatigue under repeated "
            "cyclic (traffic) loading."
        ),
        ground_truth={
            "category": "gradual_fatigue_drift",
            "cycle_period_hours": 6.0,
            "amplitude_start_mv": 0.15,
            "amplitude_growth_per_day_mv": 0.012,
            "mean_drift_per_day_mv": 0.03,
        },
    ),
    SensorProfile(
        sensor_id="beam-3b",
        location="Beam 3B",
        description="Main span beam, south side, far-shore end",
        baseline_mv=119.0,
        noise_std_mv=0.35,
        thermal_coeff_mv_per_c=0.19,
    ),
    SensorProfile(
        sensor_id="pier-cap-1",
        location="Pier Cap 1",
        description="Pier cap, near-shore support",
        baseline_mv=95.4,
        noise_std_mv=0.3,
        thermal_coeff_mv_per_c=0.15,
    ),
    SensorProfile(
        sensor_id="pier-cap-2",
        location="Pier Cap 2",
        description="Pier cap, far-shore support",
        baseline_mv=95.9,
        noise_std_mv=0.3,
        thermal_coeff_mv_per_c=0.15,
    ),
]

# Symmetric beam pairs that share a live-load signal below (see
# _generate_pair_live_load) and are therefore normally correlated. Only
# Beam 1A/1B are modeled this way -- they're the pair the scenario actually
# exercises for cross-sensor divergence; the other locations aren't claimed
# to be structurally paired, so leaving them uncorrelated keeps their own
# drift/spike detection thresholds uninflated by an unrelated shared signal.
PAIR_GROUPS: list[tuple[str, str]] = [
    ("beam-1a", "beam-1b"),
]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_pair_live_load(t_hours: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A shared traffic-load signal for one symmetric beam pair: a daily
    rush-hour-ish cycle, a weekday/weekend weekly cycle, and a slow random
    walk. Both sensors in a pair receive the *same* series on top of their
    own independent noise, which is what makes them normally correlated --
    the real-world basis for "two sensors that usually track each other"
    that cross-sensor divergence detection relies on.
    """
    phase = rng.uniform(0, 2 * np.pi)
    daily = 0.7 * np.sin(2 * np.pi * t_hours / 24 - phase)
    weekly = 0.5 * np.sin(2 * np.pi * t_hours / (24 * 7) - phase / 2)
    walk = np.cumsum(rng.normal(0, 0.02, size=len(t_hours)))
    walk -= walk.mean()
    return daily + weekly + walk


def _generate_ambient_temperature(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Shared ambient temperature series: diurnal cycle + slow random walk
    ("weather") + small measurement noise. All sensors experience the same
    ambient conditions, which is what makes thermal cross-sensitivity a
    real confound worth reasoning about.
    """
    hours = np.arange(len(timestamps))
    hour_of_day = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0

    diurnal = 7.0 * np.sin(2 * np.pi * (hour_of_day - 9) / 24)

    n_days = int(np.ceil(len(timestamps) / 24)) + 1
    daily_walk = np.cumsum(rng.normal(0, 0.6, size=n_days))
    daily_walk -= daily_walk.mean()
    weather = np.repeat(daily_walk, 24)[: len(timestamps)]

    noise = rng.normal(0, 0.4, size=len(timestamps))

    return REFERENCE_TEMP_C + diurnal + weather + noise


def generate_dataset() -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(RANDOM_SEED)

    timestamps = pd.date_range(START, END, freq=SAMPLE_INTERVAL, inclusive="left")
    t_hours = np.arange(len(timestamps), dtype=float)
    t_days = t_hours / 24.0

    temperature_c = _generate_ambient_temperature(timestamps, rng)

    pair_live_load: dict[str, np.ndarray] = {}
    for sensor_a, sensor_b in PAIR_GROUPS:
        shared = _generate_pair_live_load(t_hours, rng)
        pair_live_load[sensor_a] = shared
        pair_live_load[sensor_b] = shared

    rows = []
    for sensor in SENSORS:
        noise = rng.normal(0, sensor.noise_std_mv, size=len(timestamps))
        thermal_response = sensor.thermal_coeff_mv_per_c * (temperature_c - REFERENCE_TEMP_C)
        anomaly_offset = sensor.anomaly(t_hours, t_days) if sensor.anomaly else no_anomaly(t_hours, t_days)
        live_load = pair_live_load.get(sensor.sensor_id, np.zeros_like(t_hours))

        reading_mv = sensor.baseline_mv + thermal_response + anomaly_offset + live_load + noise

        rows.append(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "sensor_id": sensor.sensor_id,
                    "location": sensor.location,
                    "reading_mv": np.round(reading_mv, 4),
                    "temperature_c": np.round(temperature_c, 2),
                }
            )
        )

    readings = pd.concat(rows, ignore_index=True)
    readings = readings.sort_values(["timestamp", "sensor_id"]).reset_index(drop=True)

    ground_truth = {
        "note": (
            "Synthetic ground truth describing exactly how each anomaly was "
            "injected during data generation. This file is for evaluating "
            "the agent's answers after the fact -- it is not given to the "
            "agent as input."
        ),
        "random_seed": RANDOM_SEED,
        "sensors": {
            sensor.sensor_id: {
                "location": sensor.location,
                "has_injected_anomaly": sensor.anomaly is not None,
                "anomaly_note": sensor.anomaly_note or None,
                "ground_truth": sensor.ground_truth or None,
            }
            for sensor in SENSORS
        },
    }

    return readings, ground_truth


def write_dataset(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    readings, ground_truth = generate_dataset()

    readings_path = output_dir / "sensor_readings.csv"
    readings.to_csv(readings_path, index=False)

    sensors_meta = {
        "structure": STRUCTURE,
        "synthetic": True,
        "disclaimer": (
            "All data in this file and the accompanying sensor_readings.csv "
            "is synthetically generated for demo/portfolio purposes. It does "
            "not represent any real structure, sensor deployment, or "
            "measurement."
        ),
        "time_range": {
            "start": START.isoformat(),
            "end": END.isoformat(),
            "sample_interval": SAMPLE_INTERVAL,
        },
        "reference_temperature_c": REFERENCE_TEMP_C,
        "sensors": [
            {
                "sensor_id": s.sensor_id,
                "location": s.location,
                "description": s.description,
                "baseline_mv": s.baseline_mv,
            }
            for s in SENSORS
        ],
    }
    sensors_path = output_dir / "sensors.json"
    sensors_path.write_text(json.dumps(sensors_meta, indent=2), encoding="utf-8")

    ground_truth_path = output_dir / "scenario_ground_truth.json"
    ground_truth_path.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    print(f"Wrote {len(readings):,} readings for {len(SENSORS)} sensors to {readings_path}")
    print(f"Wrote sensor/structure metadata to {sensors_path}")
    print(f"Wrote scenario ground truth to {ground_truth_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    output_dir = repo_root / "data" / "sample"
    write_dataset(output_dir)


if __name__ == "__main__":
    main()
