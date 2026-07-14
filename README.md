
<div align="center">
  <br />
  <h1>Callisia: AI-Powered Clinical Wearable System</h1>
  <p>
    <b>Transforming Wearable Data into Personalized Risk Indicators for Hospital Patients</b>
  </p>
  <br />
</div>

<!-- BADGES -->
<div align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-2.x-orange.svg" alt="PyTorch Version">
</div>

---

## 📖 Overview

**Callisia** is a cutting-edge research project designed to bridge the gap between raw wearable sensor data and actionable clinical insights. By leveraging state-of-the-art Machine Learning and Large Language Models (LLMs), this project provides a robust framework for monitoring hospitalized patients, enabling early prediction of adverse events and supporting proactive clinical decision-making.

The system is built on a modular, multi-stage pipeline that ingests physiological data (PPG, ACC), extracts reliable biomarkers like Heart Rate (HR), and uses LLMs to generate interpretive clinical reports. This repository contains the complete codebase, from data analysis and model training to the final evaluation suite.

---

## ✨ Key Features

-   **End-to-End Pipeline**: A fully automated pipeline from raw sensor data to a complete clinical report.
-   **Dual-Architecture Models**: Choose between a lightweight **1D-CNN (`MultimodalHRNet`)** for on-device deployment and a high-performance **Vision Transformer (`PatchHRNet`)** for superior accuracy.
-   **LLM-Powered Insights**: Integrates with LLMs (e.g., Gemini) to provide nuanced, context-aware interpretations of physiological data, going beyond simple threshold alerts.
-   **Comprehensive Evaluation Suite**: A built-in framework to benchmark LLM providers on reliability, clinical integrity, and robustness.
-   **Reproducible Research**: The entire workflow, including data preprocessing, model training, and evaluation, is designed for reproducibility.
-   **Modular and Extensible**: The architecture is designed to be easily extended with new models, biomarkers, or evaluation metrics.

---

## 🏛️ System Architecture

The Callisia pipeline is a four-stage process designed to systematically transform raw data into a structured, interpretable clinical report.

```
[ Raw Sensor Data (PPG, ACC) ]
            │
            ▼
┌────────────────────────────────┐
│   Stage 1: Signal Processing   │
└────────────────────────────────┘
            │
            ▼
[ Continuous HR Predictions ]
            │
            ▼
┌────────────────────────────────┐
│  Stage 2: Clinical Aggregation │
└────────────────────────────────┘
            │
            ▼
[ Structured Clinical Context ]
            │
            ▼
┌────────────────────────────────┐
│      Stage 3: LLM Insights     │
└────────────────────────────────┘
            │
            ▼
[ Structured Clinical Insights ]
            │
            ▼
┌────────────────────────────────┐
│   Stage 4: Report Generation   │
└────────────────────────────────┘
            │
            ▼
[ Human-Readable Report (MD) ]
```

---

## 🔧 Core Components

### 1. Data Analysis & Model Training

This component focuses on data exploration and the creation of robust biomarker models.

-   **Exploratory Data Analysis (EDA)**: The `notebooks/` directory offers a deep dive into the datasets (Callisia, PPG-DaLiA, WESAD), covering signal characteristics, data synchronization, and movement analysis.
-   **Model Training**: The `src/models/training/` directory houses the complete training framework. We use **Leave-One-Subject-Out (LOSO)** cross-validation to ensure our models generalize to unseen individuals.

### 2. The Core Pipeline

Orchestrated by `src/orchestrator/pipeline_orchestrator.py`, this is the heart of the Callisia system.

1.  **Signal Processing**: Extracts continuous HR predictions from raw sensor data.
2.  **Clinical Aggregation**: Creates a rich, structured clinical context from the time-series data.
3.  **LLM Insights**: Feeds the context into an LLM to generate a draft clinical assessment.
4.  **Report Generation**: Formats the LLM output into a clean, human-readable Markdown report.

### 3. The Evaluation Suite

Located in `src/evaluation/`, this component provides a rigorous framework for benchmarking the LLMs used for clinical interpretation. The `BenchmarkSuite` evaluates LLM providers based on three pillars:

-   **Reliability & Consistency**: How reproducible are the LLM's outputs on the same data?
-   **Clinical Integrity**: Does the LLM's reasoning align with clinical logic? (Uses an "LLM-as-a-Judge").
-   **Decision Robustness**: How do the LLM's outputs fare when the input data is noisy or perturbed?

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.9+
-   An environment with the required dependencies (e.g., a virtual environment).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Callisia.git
    cd Callisia
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Pipeline

To run the end-to-end pipeline with a sample patient's data, execute the main script:

```bash
python main.py
```

This will run the full four-stage pipeline and generate a clinical report in the `reports/` directory.

---

## 🙏 Acknowledgments

This project is a collaborative effort between the **University of Passau** and the **Università Politecnica delle Marche**, in partnership with **Callisia S.r.l.** We thank all collaborators for their invaluable contributions.
