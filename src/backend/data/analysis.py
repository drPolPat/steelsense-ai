"""Anomaly detection pipeline for SteelSense AI.

Turns raw sensor time-series into the three structured signal categories
called out in the project brief -- gradual drift, acute spikes, and
cross-sensor divergence -- as a "tool" the agent can call. This module has
no knowledge of how the synthetic data generator injected anomalies; it
detects patterns purely from the readings themselves, the same way it
would need to on any other time series.

Method, in order:

1. Thermal correction. The first BASELINE_DAYS of each sensor's series is
   treated as a known-good commissioning baseline. A linear fit of
   reading_mv against temperature_c over that window gives a per-sensor
   thermal coefficient, which is subtracted from the full series to get a
   "residual" -- see rag/knowledge_base.py's temperature-cross-sensitivity
   entry for why this matters: without it, ordinary weather swings look
   like stress anomalies.
2. Drift detection. A linear trend fit on the post-baseline residual,
   flagged when the projected change over the window is large relative to
   the sensor's own baseline noise. Growing weekly residual variance is
   checked separately as the "cyclic loading" fatigue signature.
3. Spike detection. A rolling z-score against the baseline noise level,
   grouped into discrete events, each checked for a lasting post-event
   offset (a spike that fully recovers is a different story than one that
   doesn't).
4. Divergence detection. Each sensor's most-correlated peer is found from
   baseline-window correlations; the residual gap between a sensor and its
   peer is trend-checked for slow-onset localized divergence.

All thresholds are expressed in units of the sensor's own baseline
residual standard deviation ("sigma") so they scale sensibly across
sensors with different noise levels. They are tuned by hand against this
project's synthetic data, not derived from any standard -- see
docs/evals.md (added in a later stage) for how detection quality is
checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .ingestion import get_sensor_readings, list_sensors

BASELINE_DAYS = 14
DRIFT_SLOPE_SIGMA = 3.0
AMPLITUDE_GROWTH_MIN_RATIO = 1.5
SPIKE_Z_THRESHOLD = 5.0
SPIKE_MERGE_GAP_HOURS = 6
SPIKE_RESIDUAL_OFFSET_SIGMA = 1.5
DIVERGENCE_SLOPE_SIGMA = 3.0


@dataclass
class SensorSeries:
    sensor_id: str
    location: str
    timestamps: pd.Series
    residual_mv: np.ndarray
    baseline_std_mv: float
    thermal_coeff_mv_per_c: float


def _thermal_correct(sensor_df: pd.DataFrame) -> SensorSeries:
    baseline_cutoff = sensor_df["timestamp"].min() + pd.Timedelta(days=BASELINE_DAYS)
    baseline = sensor_df[sensor_df["timestamp"] < baseline_cutoff]

    coeff, intercept = np.polyfit(baseline["temperature_c"], baseline["reading_mv"], 1)
    predicted_thermal = coeff * sensor_df["temperature_c"] + intercept
    residual = (sensor_df["reading_mv"] - predicted_thermal).to_numpy()

    baseline_residual = residual[: len(baseline)]
    baseline_std = float(np.std(baseline_residual)) or 1e-6

    return SensorSeries(
        sensor_id=sensor_df["sensor_id"].iloc[0],
        location=sensor_df["location"].iloc[0],
        timestamps=sensor_df["timestamp"],
        residual_mv=residual,
        baseline_std_mv=baseline_std,
        thermal_coeff_mv_per_c=float(coeff),
    )


def _t_days(timestamps: pd.Series) -> np.ndarray:
    return ((timestamps - timestamps.min()).dt.total_seconds() / 86400).to_numpy()


def _detect_drift(series: SensorSeries) -> dict:
    t_days = _t_days(series.timestamps)
    post_baseline = t_days >= BASELINE_DAYS
    t, residual = t_days[post_baseline], series.residual_mv[post_baseline]

    slope_mv_per_day = float(np.polyfit(t, residual, 1)[0]) if len(t) > 1 else 0.0
    projected_change = slope_mv_per_day * (t[-1] - t[0]) if len(t) else 0.0
    drift_sigma = abs(projected_change) / series.baseline_std_mv

    daily = pd.Series(residual, index=t_days[post_baseline]).groupby(
        np.floor(t_days[post_baseline])
    ).std()
    amplitude_growth_ratio = 1.0
    amplitude_trend_mv_per_day = 0.0
    if len(daily) >= 14:
        early_std = daily.iloc[: len(daily) // 4].mean()
        late_std = daily.iloc[-len(daily) // 4 :].mean()
        if early_std and early_std > 0:
            amplitude_growth_ratio = float(late_std / early_std)
        amplitude_trend_mv_per_day = float(np.polyfit(daily.index, daily.to_numpy(), 1)[0])

    return {
        "detected": bool(drift_sigma >= DRIFT_SLOPE_SIGMA),
        "slope_mv_per_day": round(slope_mv_per_day, 4),
        "projected_change_mv": round(projected_change, 3),
        "significance_sigma": round(drift_sigma, 2),
        "growing_cyclic_amplitude": bool(amplitude_growth_ratio >= AMPLITUDE_GROWTH_MIN_RATIO),
        "amplitude_growth_ratio": round(amplitude_growth_ratio, 2),
        "amplitude_trend_mv_per_day_per_day": round(amplitude_trend_mv_per_day, 5),
    }


def _detect_spikes(series: SensorSeries) -> list[dict]:
    """Flag points that deviate sharply from their own recent (trailing)
    history, rather than from a single fixed baseline. A trailing window
    means a slow, smooth change -- like a growing cyclic-drift amplitude --
    stays within its own local stats and isn't mistaken for a spike, while
    a genuine sudden event stands out against the calmer window right
    before it.
    """
    t_days = _t_days(series.timestamps)
    residual = pd.Series(series.residual_mv)

    window = 24 * 5  # 5-day trailing window
    min_periods = 24 * 3
    rolling_mean = residual.rolling(window, min_periods=min_periods).mean().shift(1)
    rolling_std = residual.rolling(window, min_periods=min_periods).std().shift(1)
    rolling_std = rolling_std.clip(lower=series.baseline_std_mv * 0.5)

    z = ((residual - rolling_mean) / rolling_std).fillna(0.0).to_numpy()
    z[t_days < BASELINE_DAYS] = 0.0
    residual = residual.to_numpy()

    flagged_idx = np.where(np.abs(z) >= SPIKE_Z_THRESHOLD)[0]
    if len(flagged_idx) == 0:
        return []

    timestamps = series.timestamps.to_numpy()
    merge_gap = np.timedelta64(SPIKE_MERGE_GAP_HOURS, "h")

    events: list[list[int]] = [[flagged_idx[0]]]
    for idx in flagged_idx[1:]:
        if timestamps[idx] - timestamps[events[-1][-1]] <= merge_gap:
            events[-1].append(idx)
        else:
            events.append([idx])

    baseline_mean = float(np.mean(residual[t_days < BASELINE_DAYS]))
    results = []
    for event_idx in events:
        start_i, end_i = event_idx[0], event_idx[-1]
        peak_i = event_idx[int(np.argmax(np.abs(residual[event_idx])))]

        # Compare against a *settled* window well after the event (skipping
        # the exponential-decay transient) rather than the period right
        # after it, so a real but small permanent offset isn't washed out
        # by averaging over the still-decaying part of the signal.
        settle_lag_hours, settle_window_hours = 24 * 8, 24 * 5
        settled_start = min(len(residual), end_i + 1 + settle_lag_hours)
        settled_end = min(len(residual), settled_start + settle_window_hours)
        settled_window = residual[settled_start:settled_end]
        post_event_offset = float(np.mean(settled_window) - baseline_mean) if len(settled_window) else 0.0

        results.append(
            {
                "start": pd.Timestamp(timestamps[start_i]).isoformat(),
                "end": pd.Timestamp(timestamps[end_i]).isoformat(),
                "peak_mv": round(float(residual[peak_i]), 3),
                "peak_sigma": round(float(z[peak_i]), 2),
                "post_event_residual_offset_mv": round(post_event_offset, 3),
                "lasting_offset": bool(
                    abs(post_event_offset) >= SPIKE_RESIDUAL_OFFSET_SIGMA * series.baseline_std_mv
                ),
            }
        )
    return results


def _find_peer(sensor_id: str, all_series: dict[str, SensorSeries]) -> str | None:
    baseline_len = None
    correlations = {}
    for other_id, other_series in all_series.items():
        if other_id == sensor_id:
            continue
        t_days = _t_days(other_series.timestamps)
        baseline_mask = t_days < BASELINE_DAYS
        this_t_days = _t_days(all_series[sensor_id].timestamps)
        this_mask = this_t_days < BASELINE_DAYS
        n = min(baseline_mask.sum(), this_mask.sum())
        if n < 2:
            continue
        a = all_series[sensor_id].residual_mv[this_mask][:n]
        b = other_series.residual_mv[baseline_mask][:n]
        correlations[other_id] = float(np.corrcoef(a, b)[0, 1])

    if not correlations:
        return None
    return max(correlations, key=correlations.get)


def _detect_divergence(sensor_id: str, all_series: dict[str, SensorSeries]) -> dict:
    peer_id = _find_peer(sensor_id, all_series)
    if peer_id is None:
        return {"detected": False, "peer_sensor_id": None}

    series, peer = all_series[sensor_id], all_series[peer_id]
    n = min(len(series.residual_mv), len(peer.residual_mv))
    gap = series.residual_mv[:n] - peer.residual_mv[:n]
    t_days = _t_days(series.timestamps)[:n]

    post_baseline = t_days >= BASELINE_DAYS
    t, gap_post = t_days[post_baseline], gap[post_baseline]
    baseline_gap_std = float(np.std(gap[t_days < BASELINE_DAYS])) or 1e-6

    slope_mv_per_day = float(np.polyfit(t, gap_post, 1)[0]) if len(t) > 1 else 0.0
    projected_change = slope_mv_per_day * (t[-1] - t[0]) if len(t) else 0.0
    divergence_sigma = abs(projected_change) / baseline_gap_std

    return {
        "detected": bool(divergence_sigma >= DIVERGENCE_SLOPE_SIGMA),
        "peer_sensor_id": peer_id,
        "peer_location": peer.location,
        "gap_slope_mv_per_day": round(slope_mv_per_day, 4),
        "gap_projected_change_mv": round(projected_change, 3),
        "significance_sigma": round(divergence_sigma, 2),
        "note": (
            "A widening gap is inherently symmetric -- it shows this pair is "
            "diverging, not which sensor caused it. That requires a third "
            "reference (the wider sensor network, history, or inspection)."
        ) if divergence_sigma >= DIVERGENCE_SLOPE_SIGMA else None,
    }


def _to_jsonable(value):
    """Recursively convert numpy scalar types to plain Python types so the
    result can be json-dumped as a tool result. round()/np.polyfit() leave
    numpy float64/bool_ scalars in place even after arithmetic with Python
    floats, which json.dumps rejects."""
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_all_series() -> dict[str, SensorSeries]:
    return {
        sensor["sensor_id"]: _thermal_correct(get_sensor_readings(sensor["sensor_id"]))
        for sensor in list_sensors()
    }


def analyze_sensor(sensor_id: str, all_series: dict[str, SensorSeries] | None = None) -> dict:
    """Run the full drift / spike / divergence pipeline for one sensor and
    return a structured finding -- the shape the agent's tool hands to the
    LLM."""
    all_series = all_series if all_series is not None else _load_all_series()
    if sensor_id not in all_series:
        raise ValueError(f"Unknown sensor_id: {sensor_id!r}")
    series = all_series[sensor_id]

    drift = _detect_drift(series)
    spikes = _detect_spikes(series)
    divergence = _detect_divergence(sensor_id, all_series)

    flags = []
    if drift["detected"]:
        flags.append("gradual_fatigue_drift")
    if drift["growing_cyclic_amplitude"]:
        flags.append("growing_cyclic_amplitude")
    if any(s["lasting_offset"] for s in spikes):
        flags.append("acute_stress_spike_with_residual")
    elif spikes:
        flags.append("acute_stress_spike")
    if divergence["detected"]:
        flags.append("cross_sensor_divergence")

    concern_score = max(
        drift["significance_sigma"],
        max((s["peak_sigma"] for s in spikes), default=0.0),
        divergence["significance_sigma"],
    )

    return _to_jsonable(
        {
            "sensor_id": sensor_id,
            "location": series.location,
            "thermal_coefficient_mv_per_c": round(series.thermal_coeff_mv_per_c, 4),
            "baseline_noise_std_mv": round(series.baseline_std_mv, 4),
            "drift": drift,
            "spikes": spikes,
            "divergence": divergence,
            "flags": flags,
            "concern_score": round(concern_score, 2),
        }
    )


def analyze_all_sensors() -> dict:
    """Run analyze_sensor for every sensor and rank by concern_score --
    backs the "which sensor shows the most concerning trend" style query."""
    all_series = _load_all_series()
    findings = {sid: analyze_sensor(sid, all_series) for sid in all_series}
    ranking = sorted(findings.values(), key=lambda f: f["concern_score"], reverse=True)
    return {
        "sensors": findings,
        "ranked_by_concern": [
            {"sensor_id": f["sensor_id"], "location": f["location"], "concern_score": f["concern_score"], "flags": f["flags"]}
            for f in ranking
        ],
    }
