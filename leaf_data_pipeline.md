# Clinical Wearable Insights Pipeline
**Combining Traditional Machine Learning and Large Language Models for Healthcare: A Hybrid Approach for Interpreting Wearable Data**

## Overview
This repository contains a robust, production-grade signal processing and data aggregation pipeline. It is designed to ingest raw, high-frequency physiological data from wearable sensors (Blood Volume Pulse and Accelerometer), harmonize hardware inconsistencies, execute predictive Machine Learning models for continuous heart rate, and transform the kinetic/cardiovascular results into a highly structured JSON payload optimized for Large Language Model (LLM) context generation.

## System Architecture
The architecture is divided into two primary orchestrators:

### 1. `SignalProcessingPipeline` (Data Ingestion & Inference)
Handles the "Raw to Predict" phase. Wearable hardware often suffers from timing drift, jitter, and dropped packets. This module acts as the clinical data cleaner.
* **Temporal Harmonization:** Converts raw integer-based milliseconds into time-aware Pandas indices.
* **Strict Grid Snapping:** Resamples asymmetrical sensor inputs onto a strict, unified mathematical grid (e.g., 40Hz).
* **Signal Bridging:** Applies linear interpolation to bridge micro-dropouts (up to 1 second) to maintain signal continuity, while actively pruning massive gaps that are clinically unreliable.
* **Inference Orchestration:** Passes the perfectly synchronized BVP and ACC arrays to the neural network (`HRPredictor`) to generate continuous heart rate predictions.

### 2. `ClinicalAggregator` (Feature Extraction & LLM Packaging)
Handles the "Predict to Context" phase. It transforms the continuous high-frequency arrays into macroscopic, human-readable clinical summaries using 8-second rolling windows.
* **Cardiovascular Statistics:** Calculates median, variance, and baseline percentiles, and categorizes beats into clinical zones (Bradycardia, Normal, Tachycardia).
* **Kinetic/Motion Profiling:** Extracts multi-dimensional features to accurately classify physical exertion and detect motion artifacts that corrupt PPG sensors.
* **Autonomic Nervous System Proxy:** Measures beat-to-beat volatility to estimate physiological rigidity vs. dynamism.
* **Clinical Contextualization:** Computes Pearson correlation coefficients to determine if a patient's elevated heart rate is justified by physical activity or indicates an isolated anomaly (e.g., psychological stress, arrhythmia).

## Methodology: Multi-Metric Motion Tracking
To prevent the LLM from receiving "false positive" cardiovascular alarms caused by sensor noise, the movement analysis relies on an interlocking, multi-dimensional thresholding system:

1. **Standard Deviation (`std_magnitude`):** Captures continuous, rhythmic energy expenditure (e.g., walking, jogging). Used to directly correlate physical effort with heart rate.
2. **Peak-to-Peak Range (`range_magnitude`):** The "Impact Alarm." Captures sudden, violent movements (e.g., table smacks, sudden arm swings) that average out in variance calculations. Also acts as the primary heuristic for detecting sudden jolts, potential falls, or dropped sensors.
3. **Mean Absolute Difference (`mean_jerk`):** The "Jitter Alarm." Captures high-frequency micro-movements with low physical distance but rapid directional changes (e.g., shivering, typing, nervous tapping).

By combining these metrics via logical `AND/OR` gates, the pipeline strictly categorizes patient states into `RESTING`, `LIGHT_MOVEMENT`, and `ACTIVE` zones.

## Output Structure
The final output of the pipeline is a strict, structured payload designed specifically for an LLM's system prompt. Here's a small and not complete example of the JSON output that the `ClinicalAggregator` would produce after processing a 2-minute recording:

```json
{
  "system_telemetry": {
    "total_recording_duration_seconds": 120,
    "rolling_window_duration_seconds": 8,
    "rolling_window_step_seconds": 2,
    "data_alignment_note": "..."
  },
  "cardiovascular_analysis": {
    "standard_statistics": {"mean_hr": 72.4, "variance_hr": 14.2},
    "trend": "stable",
    "beats_distribution": {"under_60_bpm": 0, "between_60_and_100_bpm": 150, "over_100_bpm": 0}
  },
  "movement_analysis": {
    "mean_noise_level": 12.3,
    "sudden_jolt_detected": false,
    "distribution": {
      "resting_windows_count": 45,
      "light_movement_windows_count": 12,
      "active_movement_windows_count": 3
    }
  },
  "clinical_context": {
    "hr_movement_correlation": 0.82,
    "clinical_context": "Heart rate strongly driven by physical activity."
  }
}
```