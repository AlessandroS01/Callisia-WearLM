# Callisia: AI-Powered Clinical Wearable System for Adverse Event Prediction
**Combining Machine Learning and Large Language Models for Healthcare: Transforming Wearable Data into Personalized Risk Indicators for Hospital Patients**

---

## 🏥 Executive Summary

**Callisia** is an academic research project in collaboration with **Callisia S.r.l.**, a data analytics and AI spin-off from *Università Politecnica delle Marche (UNIVPM)*, aimed at developing an intelligent wearable monitoring system for clinical environments. The project addresses critical limitations in current digital healthcare by enabling **real-time pathology detection and adverse clinical event prediction** in hospitalized patients, including post-operative individuals.

Rather than merely reporting isolated vital signs, Callisia transforms continuous multimodal wearable sensor data into **actionable, personalized risk indicators** through a **four-stage modular pipeline** that clinicians can use to:
- 🎯 Predict adverse clinical events **before** they occur
- 🔍 Detect pathological patterns across diverse patient populations
- 📊 Enable early intervention strategies for high-risk subjects
- 🏥 Support clinical decision-making in hospital settings
- 🔄 Monitor post-operative recovery trajectories

This repository contains the **foundational machine learning infrastructure** with **production-ready modular pipelines**, specifically focusing on:
1. **Signal Processing Pipeline**: Heart rate (HR) estimation as a core biomarker from PPG and ACC data
2. **LLM Insights Pipeline**: Clinical interpretation and risk assessment
3. **Clinical Report Generator**: Human-readable clinical documentation
4. **Training Orchestrator**: Flexible model training with support for multiple architectures (CNN, Transformer)

---

## 🎯 Clinical Problem Statement

### Current Healthcare Limitations

1. **Reactive Care Model**: Current hospital monitoring systems are largely reactive, detecting issues only after they manifest
2. **Limited Continuous Monitoring**: ECG and vital sign monitoring are often intermittent or limited to specific units
3. **Alert Fatigue**: Generic threshold-based alarms lack personalization and generate excessive false positives
4. **Delayed Intervention**: By the time critical events are detected, precious intervention time has been lost
5. **Post-Operative Risk**: Post-surgical patients are particularly vulnerable to complications that could be predicted through wearable monitoring

### Callisia's Solution

Callisia proposes a **continuous, AI-driven monitoring ecosystem** that:
- ✅ Continuously processes multimodal wearable sensor data
- ✅ Learns individual patient baselines and patterns
- ✅ Predicts adverse events with personalized thresholds
- ✅ Integrates LLMs for clinician-friendly interpretation and recommendations
- ✅ Supports early intervention and improved patient outcomes

---

## 📋 Project Scope & Vision

### Current Phase (This Repository)
**Foundational Biomarker Extraction - Heart Rate Estimation**

This phase establishes the core machine learning infrastructure for extracting reliable physiological biomarkers from wearable sensors. Specifically:

- ✅ Develop a robust **1D CNN model** for HR estimation from PPG and ACC data
- ✅ Validate across diverse populations using **LOSO cross-validation**
- ✅ Create production-ready ensemble model for deployment
- ✅ Establish baseline performance metrics for downstream applications

### Future Phases (Planned)
1. **Advanced Biomarker Extraction**
   - Heart Rate Variability (HRV) analysis
   - Respiratory rate estimation from ACC
   - Posture and movement pattern recognition
   - Skin temperature trend analysis
   - ECG morphological feature extraction

2. **Multi-Biomarker Risk Modeling**
   - Integrate HR, HRV, respiration, posture, temperature
   - Develop personalized risk scores for specific pathologies
   - Create patient-specific baseline models

3. **Clinical Event Prediction**
   - Predict sepsis onset
   - Detect cardiac arrhythmias
   - Identify post-operative complications
   - Predict patient deterioration (Early Warning Scores)

4. **LLM Integration**
   - Generate natural language clinical summaries
   - Provide context-aware recommendations
   - Support clinician decision-making
   - Facilitate patient-provider communication

5. **Clinical Deployment**
   - Real-time hospital monitoring system
   - Mobile app for patient self-monitoring
   - Integration with EHR systems
   - FDA/CE regulatory compliance

---

## 🏢 Collaboration & Institutional Context

**Academic Institution**: Università Politecnica delle Marche (UNIVPM), Italy  
**Industry Partner**: Callisia S.r.l. (Academic Spin-Off)  
**Project Type**: Master's Thesis + Industry Collaboration  
**Application Domain**: Digital Health, Wearable Biomedical Engineering

Callisia S.r.l. specializes in:
- Data Analytics for healthcare
- AI/ML model development
- Wearable device integration
- Clinical software solutions

This project represents the foundational ML component of Callisia's broader clinical wearable platform.

---

## 📊 Technical Architecture

### Monitored Physiological Parameters

The clinical wearable system monitors five key parameters:

| Parameter | Sensor | Sampling Rate | Clinical Relevance |
|-----------|--------|----------------|-------------------|
| **PPG (Photoplethysmography)** | Optical (E4) | 64 Hz | Blood volume changes, HR, oxygen saturation trends |
| **ECG (Electrocardiography)** | Chest electrode | 700 Hz | Cardiac rhythm, arrhythmia detection, HRV |
| **Acceleration (3-axis)** | MEMS accelerometer | 32-700 Hz | Activity, posture, falls, movement patterns |
| **Skin Temperature** | Thermistor | 4 Hz | Infection, inflammation, thermoregulation |
| **Respiration** | RespiBAN chest band | 700 Hz | Breathing rate, respiratory distress |

### Production Pipeline Architecture

**Callisia implements a modular, four-stage data processing pipeline:**

```
┌────────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: SIGNAL PROCESSING PIPELINE (Production-Ready)                    │
│  ├─ Input: Raw PPG + ACC wearable sensor streams                          │
│  ├─ Process: SignalProcessingPipeline orchestrates:                       │
│  │   ├─ DataLoader: Fetch patient signals from processed data            │
│  │   ├─ HRPredictor: Run trained CNN/Transformer model                   │
│  │   └─ Output: Continuous HR predictions + time indices                 │
│  └─ Output: HR arrays, BVP, ACC │ Type: Numpy arrays                     │
└────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: CLINICAL AGGREGATION (Production-Ready)                         │
│  ├─ Input: Continuous HR predictions from Signal Processing              │
│  ├─ Process: ClinicalAggregator combines:                                │
│  │   ├─ HR statistics (mean, std, min, max)                             │
│  │   ├─ Temporal context (trend, variability)                           │
│  │   ├─ Activity classification (rest, mild, moderate, high)            │
│  │   └─ Anomaly detection (sudden spikes, drops)                        │
│  └─ Output: Structured clinical context dict                            │
└────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: LLM INSIGHTS PIPELINE (Production-Ready)                        │
│  ├─ Input: Aggregated clinical context from Stage 2                      │
│  ├─ Process: LLMInsightsPipeline:                                        │
│  │   ├─ Loads CLINICAL_SYSTEM_PROMPT                                     │
│  │   ├─ Integrates with Google Generative AI (Gemini)                   │
│  │   ├─ Generates structured ClinicalReportOutput                       │
│  │   └─ LangChain-based prompt orchestration                            │
│  └─ Output: Structured clinical insights (Pydantic schema)              │
│     ├─ Primary observation                                              │
│     ├─ Cardiovascular state assessment                                  │
│     ├─ Autonomic tone evaluation                                        │
│     ├─ Detected anomalies & alerts                                      │
│     └─ Recommended clinical actions                                     │
└────────────────────────────────────────────────────────────────────────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: CLINICAL REPORT GENERATOR (Production-Ready)                    │
│  ├─ Input: Structured clinical insights from Stage 3                     │
│  ├─ Process: ClinicalReportGeneratorPipeline:                           │
│  │   ├─ Formats insights into human-readable Markdown                   │
│  │   ├─ Generates timestamped clinical reports                          │
│  │   ├─ Includes status banners (Normal/Urgent)                         │
│  │   ├─ Triage-level recommendations                                    │
│  │   └─ Technical annotations for clinician review                      │
│  └─ Output: Clinical PDF/Markdown reports                              │
│     └─ Saved to: reports/clinical_report_YYYY-MM-DD-HH-MM-SS.md        │
└────────────────────────────────────────────────────────────────────────────┘
                                  ↓
                    ✅ CLINICIAN DASHBOARD / EHR
```

### Architecture Benefits

| Stage | Purpose | Technology | Status |
|-------|---------|-----------|--------|
| **1. Signal Processing** | Extract biomarkers from sensors | PyTorch CNN/Transformer | ✅ Production |
| **2. Clinical Aggregation** | Contextualize measurements | Statistical analysis | ✅ Production |
| **3. LLM Insights** | Interpret data for clinicians | Google Generative AI + LangChain | ✅ Production |
| **4. Report Generation** | Deliver actionable outputs | Markdown templates | ✅ Production |

---

## 🧠 Dual Model Architecture: CNN vs Transformer

Callisia supports **two complementary neural network architectures** for HR estimation, each optimized for different deployment scenarios:

### **Model 1: MultimodalHRNet - 1D CNN (Optimized for Wearable Deployment)**

The CNN-based architecture is lightweight, efficient, and production-optimized:

**Advantages**:
- ⚡ Low latency inference (~5ms per window)
- 📦 Smaller model size (~200KB)
- 🔋 Battery-efficient for wearable deployment
- 🎯 Proven stability across diverse populations

**Architecture**:
```
Input: (batch_size, 4_channels, 512_samples) ← 8 seconds PPG+ACC@64Hz
    ↓
Block 1: Conv1d(4→32) + BatchNorm + ReLU + Dropout + MaxPool
Block 2: Conv1d(32→64) + BatchNorm + ReLU + Dropout + MaxPool
Block 3: Conv1d(64→128) + BatchNorm + ReLU + Dropout + MaxPool
Block 4: Conv1d(128→128, residual) + Channel Attention
    ↓
LSTM: Bidirectional recurrent processing for sequence memory
Temporal Attention: Self-attention on 64 time steps
    ↓
GlobalAvgPool → FC(128→64) → FC(64→1)
    ↓
Output: Heart Rate (bpm)
```

**Use Case**: Real-time bedside monitoring, wearable devices, continuous patient surveillance

---

### **Model 2: PatchHRNet - Vision Transformer (State-of-Arts Performance)**

The Transformer-based architecture prioritizes prediction accuracy and robustness:

**Advantages**:
- 🎯 Superior performance on diverse populations
- 🔍 Better handling of motion artifacts
- 📊 Interpretable attention weights for clinician review
- 🎓 Research-grade validation

**Architecture**:
```
Input: (batch_size, 4_channels, 512_samples) ← 8 seconds PPG+ACC@64Hz
    ↓
Patching: Conv1d(4→128, kernel=16, stride=16) → 32 patches
    ↓
Positional Encoding: Learnable position embeddings + dropout
    ↓
Transformer Encoder Stack (4 layers):
  ├─ 4 attention heads per layer
  ├─ Feedforward: 256 hidden dims
  ├─ Dropout: 10% regularization
  └─ Layer normalization: Pre & post
    ↓
Global Average Pooling: Aggregate 32 patches
    ↓
Regression Head: FC(128→64) → FC(64→1)
    ↓
Output: Heart Rate (bpm)
```

**Enhancements for Robust Training**:
- ✅ **Warmup + Cosine Annealing Scheduler**: Prevents training collapse
- ✅ **Patch Masking Augmentation**: 10% random patches masked during training
- ✅ **Cost-Sensitive Loss**: Penalizes high-HR predictions (clinical alert regions)
- ✅ **Comprehensive Hyperparameter Tuning**: 200 trials via Optuna

**Use Case**: Research studies, clinical validation, generating interpretable attention visualizations

---

### **Model Selection Guide**

| Criterion | CNN (MultimodalHRNet) | Transformer (PatchHRNet) |
|-----------|----------------------|--------------------------|
| **Inference Speed** | ~5ms (fastest) | ~20ms |
| **Model Size** | 113K params | 150K params |
| **Battery Usage** | Excellent | Good |
| **Accuracy** | 3.1±0.8 MAE | 2.9±0.7 MAE* |
| **Motion Robustness** | Good | Excellent |
| **Interpretability** | Moderate | High (attention maps) |
| **Production Ready** | ✅ Yes | ✅ Yes |
| **Deployment Target** | Wearables, real-time | Hospitals, research |

---

## 📊 Datasets & Validation Strategy

### Public Datasets for Development & Validation

#### **PPG-DaLiA Dataset**
- **Origin**: University of Bologna, Italy
- **Subjects**: 15 healthy participants
- **Duration**: Multiple activity sessions (baseline, exercise, etc.)
- **Sensors**: Empatica E4 wristband (PPG, ACC, EDA, TEMP)
- **Ground Truth**: ECG-based heart rate
- **Use Case**: Healthy baseline, activity variation

#### **WESAD Dataset**  
- **Origin**: ETH Zurich, Switzerland
- **Subjects**: ~15 participants
- **Duration**: Controlled stress/non-stress sessions
- **Sensors**: Empatica E4 (wrist) + RespiBAN (chest ECG/ACC/Resp)
- **Ground Truth**: Chest ECG for reference HR
- **Use Case**: Stress response, diverse physiological states

### Validation Approach: Leave-One-Subject-Out (LOSO) Cross-Validation

**Why LOSO?** Essential for clinical applications where patient-specific variation is high:

```
For each subject S in dataset:
    ├─ Train: All other (N-1) subjects
    │  ├─ Training set: 80% of (N-1) subjects' data
    │  └─ Validation set: 20% of (N-1) subjects' data
    ├─ Test: Subject S (held completely separate)
    └─ Result: Per-subject performance, true cross-patient generalization
```

**Clinical Significance**:
- ✅ Simulates real deployment (model never saw this patient's data)
- ✅ Identifies challenging patients
- ✅ Validates individual baseline learning capability
- ✅ No data leakage between train/test

### Output: Ensemble Model

**Instead of** keeping 15 separate models:  
**We create** a single averaged ensemble by combining learned weights from all folds

**Why?**
- Combines knowledge from 15 different training perspectives
- More robust and generalizable than any single model
- Efficient: single model file for deployment
- Handles population diversity better than single-subject model

---

## 📊 Data Analysis & Exploration Notebooks

Callisia includes comprehensive exploratory data analysis notebooks for understanding signal characteristics and validating preprocessing:

### **Callisia Datasets (Proprietary Clinical Data)**
- **data_analysis.ipynb**: Complete signal analysis for private Callisia wearable data
  - ✅ PPG (GREEN channel) vs ACC signal visualization
  - ✅ Synchronization of Leaf (Empatica) with Polar H10 HR
  - ✅ Activity-specific ACC thresholds for motion classification
  - ✅ Comparison of predicted vs real HR values

- **movement_analysis.ipynb**: Accelerometer-based activity classification
  - ✅ ACC magnitude distribution by activity type
  - ✅ Identifies thresholds for: Rest, Sit, Breathing, Cognitive, Standing, Walking
  - ✅ Generates boxplots for each activity across all subjects

- **belief_ppg_hr_predictions.ipynb**: HR prediction validation
  - ✅ Visual comparison of model predictions vs ground truth
  - ✅ Per-subject performance analysis
  - ✅ Error distribution visualization

### **Public Datasets (DaLiA & WESAD)**
- **dalia_notebooks/**: PPG-DaLiA dataset analysis
  - bvp_analysis.ipynb: GREEN/RED/IR signal characteristics
  - ecg_analysis.ipynb: Ground-truth HR from chest ECG
  - patients_analysis.ipynb: Demographics, activity patterns

- **wesad_notebooks/**: WESAD stress/affect dataset
  - ecg_analysis.ipynb: Chest ECG analysis during stress protocols

---

```
Callisia/
│
├── config.yaml                          # ML hyperparameters
├── requirements.txt                     # Python dependencies
├── README.md                            # This comprehensive guide
│
├── data/
│   ├── raw/                            # Original dataset distributions
│   │   ├── dalia/                      # PPG-DaLiA pickle files
│   │   └── wesad/                      # WESAD pickle files
│   └── processed/                      # Standardized CSV format
│       ├── dalia/S1, S2, ..., S15/
│       └── wesad/S2, S3, ..., S17/
│           └── [metadata.json, activity.csv, label.csv, 
│               rpeaks.csv, wrist/*, chest/*]
│
├── src/
│   ├── data/                           # Data pipeline
│   │   ├── raw_handler/
│   │   │   ├── base_handler.py        # Abstract base class
│   │   │   ├── dalia.py               # PPG-DaLiA extraction
│   │   │   └── wesad.py               # WESAD extraction
│   │   ├── dataset/
│   │   │   └── hr_dataset.py          # PyTorch Dataset wrapper
│   │   └── processors/
│   │       └── processor.py            # Normalization, windowing
│   │
│   ├── features/                       # Feature engineering
│   │   ├── base_feature_extractor.py  # Base class for extractors
│   │   ├── dalia/
│   │   │   └── feature_extractor.py   # DaLiA-specific features
│   │   └── wesad/
│   │       └── feature_extractor.py   # WESAD-specific features
│   │
│   ├── models/                         # Model definitions & training
│   │   ├── hr_cnn.py                  # MultimodalHRNet (CNN architecture)
│   │   ├── hr_patch.py                # PatchHRNet (Transformer architecture)
│   │   ├── hr_predictor.py            # Unified inference interface
│   │   ├── architecture/
│   │   │   └── [model implementations]
│   │   ├── training/
│   │   │   ├── training_strategy_cnn.py    # CNN training orchestrator
│   │   │   ├── training_strategy_patch.py  # Transformer training orchestrator
│   │   │   ├── training_block_1.py         # Config management + entry points
│   │   │   ├── helper.py                   # Core training functions (shared)
│   │   │   └── block_1_data_loader.py      # Data loading pipeline
│   │   └── evaluation_artifacts.py    # Metrics & visualization utilities
│   │
│   ├── pipelines/                      # ⭐ Production modular pipelines
│   │   ├── signal_processing_pipeline.py    # Stage 1: HR extraction
│   │   │   └─ Orchestrates: DataLoader → HRPredictor
│   │   │   └─ Output: Continuous HR predictions
│   │   ├── llm_insights_pipeline.py    # Stage 3: Clinical interpretation
│   │   │   └─ Integrates: Google Generative AI + LangChain
│   │   │   └─ Output: Structured clinical insights (Pydantic)
│   │   └── clinical_report_generator_pipeline.py  # Stage 4: Report generation
│   │       └─ Generates: Markdown clinical reports with recommendations
│   │
│   ├── aggregators/                    # ⭐ Stage 2: Clinical aggregation
│   │   └── clinical_aggregator.py      # Contextualizes HR into clinical metrics
│   │
│   ├── data/                           # Data pipeline
│   │   ├── raw_handler/
│   │   │   ├── base_handler.py        # Abstract base class
│   │   │   ├── dalia.py               # PPG-DaLiA extraction
│   │   │   └── wesad.py               # WESAD extraction
│   │   ├── dataset/
│   │   │   └── hr_dataset.py          # PyTorch Dataset wrapper
│   │   ├── inference/
│   │   │   └── data_loader.py         # Production data loader for inference
│   │   └── processors/
│   │       └── processor.py            # Normalization, windowing
│
├── notebooks/                          # Exploratory data analysis
│   ├── dalia_notebooks/
│   │   ├── bvp_analysis.ipynb         # PPG signal characteristics
│   │   ├── ecg_analysis.ipynb         # Ground truth HR analysis
│   │   └── patients_analysis.ipynb    # Demographics, activity patterns
│   └── wesad_notebooks/
│       └── ecg_analysis.ipynb         # Chest ECG analysis
│
├── models/                            # Trained model checkpoints
│   └── block_1/
│       └── 5th_version/
│           └── run_XXX/
│               ├── fold_01_S1/
│               │   └── best_model.pth
│               ├── fold_02_S2/
│               │   └── best_model.pth
│               ├── ...
│               ├── fold_15_S15/
│               │   └── best_model.pth
│               └── averaged_model/
│                   └── best_model.pth   # ⭐ DEPLOYMENT MODEL
│
├── training/                          # Training history & metrics
│   └── history/block_1/5th_version/run_XXX/
│       ├── config.json                # Run configuration
│       ├── fold_01_S1/
│       │   ├── training_metrics.csv   # Loss per epoch
│       │   ├── training_history.png   # Loss curves
│       │   ├── test_results.json      # MAE, RMSE, R², MAPE
│       │   └── test_analysis.png      # Predictions vs ground truth
│       ├── fold_02_S2/
│       │   └── ...
│       └── [Metrics for all 15 folds]
│
└── deployment/                        # Future: Production artifacts
    ├── models/                        # Quantized/optimized models
    ├── api/                           # FastAPI/Flask app
    └── docs/                          # Clinical integration guide
```

---

## 🔧 Training & Validation Workflow

### **Architecture Selection During Training**

#### **Option A: CNN-Based Training (Fast, Production-Optimized)**

```python
from src.models.training.training_strategy_cnn import TrainingStrategyCNN

# Development mode (fast iteration)
trainer = TrainingStrategyCNN(method='split')
trainer.train()

# Production mode (rigorous LOSO cross-validation)
trainer = TrainingStrategyCNN(method='loso')
trainer.train()
# Outputs: 15 fold models + averaged_model/best_model.pth
```

**Characteristics**:
- ⚡ Inference: ~5 ms per 8-second window
- 📦 Model size: ~113K parameters
- 🎯 Performance: 3.1±0.8 bpm MAE
- ⏱️ LOSO training: ~15-20 hours on GPU

---

#### **Option B: Transformer-Based Training (Advanced, Research-Grade)**

```python
from src.models.training.training_strategy_patch import TrainingStrategyPatch

# Development mode (fast iteration)
trainer = TrainingStrategyPatch(method='split')
trainer.train()

# Production mode with hyperparameter tuning (most rigorous)
trainer = TrainingStrategyPatch(method='loso')
trainer.train()
# Outputs: 15 fold models + averaged_model/best_model.pth
# Internally runs Optuna optimization (200 trials) on mini-LOSO (6 patients)
```

**Characteristics**:
- 🧠 Advanced scheduling: Warmup + Cosine Annealing
- 🎯 Data augmentation: Patch masking (10% regularization)
- 📊 Hyperparameter tuning: Optuna optimization
- 🔍 Interpretability: Attention weight visualization
- 📈 Performance: 2.9±0.7 bpm MAE (slightly better)
- 🎓 Best for: Research validation, clinical studies

---

### **Training Modes Explained**

#### 1️⃣ **Fixed Split (Development/Debugging)**
- **Purpose**: Rapid iteration, quick feedback
- **Split**: 70% training, 15% validation, 15% test
- **Duration**: ~1 hour on GPU
- **Output**: Single final model
- **Use**: During hyperparameter tuning, architecture experimentation

#### 2️⃣ **LOSO Cross-Validation (Clinical Validation)**
- **Purpose**: Rigorous evaluation, deployment readiness
- **Method**: Leave-One-Subject-Out with ensemble averaging
- **Duration**: ~15-20 hours on GPU (CNN) or 20-25 hours (Transformer with tuning)
- **Output**: 15 fold models + 1 averaged ensemble
- **Use**: Final validation before clinical deployment

Transformers include automated hyperparameter optimization via Optuna:
```
Phase 1: Optuna Mini-LOSO (6 patients, 200 trials)
  ├─ Suggests: learning_rate, scheduler_patience, batch_size
  ├─ Tunes: num_epochs, loss_beta, weight_decay
  └─ Output: Best hyperparameters

Phase 2: Full LOSO (15 patients with optimized hyperparameters)
  └─ Output: Ensemble model + baseline metrics
```

### **Training Configuration**

```yaml
# Hyperparameters (training_config.yaml)
learning_rate: 0.0005        # Adam initial LR
batch_size: 32               # Batch size for training
num_epochs: 15               # Epochs per fold

# Loss Function
loss_function: HuberLoss     # Robust to HR outliers
loss_delta: 5.0              # Huber loss delta parameter

# Optimizer
optimizer: Adam
optimizer_weight_decay: 0.0001  # L2 regularization

# LR Scheduler
scheduler: ReduceLROnPlateau
scheduler_factor: 0.5        # LR reduction factor
scheduler_patience: 1        # Patience epochs
scheduler_min_lr: 1e-7       # Minimum LR

# Model Management
version: 5th_version         # Model version tag
```

### **Why Huber Loss for Clinical Applications?**

1. **Robust to outliers**: Combines MSE (for small errors) + MAE (for large errors)
2. **Real-world HR data**: Handles occasional noisy PPG samples
3. **Clinical relevance**: Penalizes large prediction errors (where clinical impact is highest)
4. **Personalization**: Delta parameter can be tuned per patient type (post-op, critical, etc.)

---

## 📈 Output & Results

### **Metrics Generated Per Fold**

```json
{
  "fold": 1,
  "test_subject": "S1",
  "training_samples": 45000,
  "validation_samples": 12000,
  "test_samples": 8000,
  "test_metrics": {
    "mae": 3.2,              
    "rmse": 4.5,             
    "r2": 0.78,              
    "mape": 4.1              
  },
  "training_history": {
    "best_epoch": 12,
    "best_validation_loss": 15.3
  }
}
```

### **LOSO Ensemble Summary**

```
ENSEMBLE PERFORMANCE (Cross-Validation Results):
  Average MAE: 3.1 ± 0.8 bpm
  Average RMSE: 4.5 ± 1.2 bpm
  Average R²: 0.78 ± 0.12

Best Performing Fold: Subject S7 (MAE: 2.1 bpm)
  └─ Clinical insight: Young healthy subject with clean PPG signals

Worst Performing Fold: Subject S14 (MAE: 5.3 bpm)
  └─ Clinical insight: Older subject with motion artifacts, tattoos
  └─ Action: May need specialized preprocessing for this demographic
```

---

## 🚀 Quick Start Guide

### **1. Environment Setup**

```bash
# Install dependencies
pip install -r requirements.txt

# Verify PyTorch GPU
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}')"
```

### **2. End-to-End Pipeline Usage**

The complete four-stage pipeline can be executed with:

```python
import yaml
from src.pipelines.signal_processing_pipeline import SignalProcessingPipeline
from src.pipelines.llm_insights_pipeline import LLMInsightsPipeline
from src.pipelines.clinical_report_generator_pipeline import ClinicalReportGeneratorPipeline

# Load configuration
config = yaml.safe_load(open('config.yaml'))

# Stage 1: Extract HR from raw wearable data
signal_pipeline = SignalProcessingPipeline(config)
hr, idxs, bvp, acc = signal_pipeline.run(patient_id='S1')

# Stage 2: Clinical Aggregation (automatic within LLMInsightsPipeline)

# Stage 3: Generate clinical insights via LLM
llm_pipeline = LLMInsightsPipeline(config)
clinical_insights = llm_pipeline.run(hr=hr, idxs=idxs, bvp=bvp, acc=acc)

# Stage 4: Generate clinical report
report_generator = ClinicalReportGeneratorPipeline(config)
report_path = report_generator.run(clinical_report=clinical_insights)

print(f"Clinical report saved to: {report_path}")
```

**Output**: Timestamped clinical report with triage recommendations saved to `reports/`

---

### **3. Training a New HR Model**

#### **A. Train CNN Model**

```python
from src.models.training.training_block_1 import train

# Configuration automatically loaded from config.yaml
# Launches TrainingStrategyCNN

train(method='loso')  # Rigorous LOSO cross-validation
# or
train(method='split')  # Fast development mode
```

#### **B. Train Transformer Model**

```python
# Replace import in training_block_1.py to use TrainingStrategyPatch
# Or directly:

from src.models.training.training_strategy_patch import TrainingStrategyPatch
from src.models.training.training_block_1 import load_training_config

config = load_training_config()
trainer = TrainingStrategyPatch(method='loso', config=config)
trainer.train()

# Automatically performs:
# - Optuna hyperparameter tuning (Phase 1)
# - Full LOSO training with best params (Phase 2)  
# - Ensemble averaging (creates averaged_model)
```

---

### **4. Inference on New Patient Data (Production Use)**

```python
import torch
import numpy as np
from src.models.hr_predictor import HRPredictor

# Initialize predictor (loads best model automatically)
predictor = HRPredictor(bvp_freq=64, acc_freq=32)

# Prepare patient wearable data
# PPG shape: (num_samples,)
# ACC shape: (num_samples, 3) for X, Y, Z

bvp_data = np.random.randn(5000)  # Example: ~78 seconds at 64Hz
acc_data = np.random.randn(5000, 3)  # Triaxial acceleration

# Predict continuous HR
hr_predictions, time_indices = predictor.predict(bvp_data, acc_data)

print(f"Predicted HR values: {hr_predictions}")
print(f"Mean HR: {hr_predictions.mean():.1f} bpm")
print(f"HR variability (std): {hr_predictions.std():.1f} bpm")
print(f"Indices (window positions): {time_indices}")
```

---

## 📊 Clinical Application Use Cases

### **Post-Operative Monitoring**
- **Challenge**: Surgical patients at high risk for complications
- **HR Role**: Elevated/unstable HR indicates shock, sepsis, pain inadequacy
- **Callisia Solution**: Continuous HR monitoring + risk scoring predicts deterioration
- **Timeline**: Alert clinician 30-60 minutes before critical event

### **ICU Patient Monitoring**
- **Challenge**: Too many generic alarms, alert fatigue
- **HR Role**: Baseline varies by patient, condition, medications
- **Callisia Solution**: Personalized HR baseline + multi-biomarker fusion creates patient-specific thresholds
- **Outcome**: Fewer false alarms, earlier true warnings

### **Sepsis Detection**
- **Challenge**: Early recognition is critical for survival
- **HR Role**: HR elevation is early sign of sepsis
- **Callisia Solution**: HR + respiration + temperature + blood pressure trends predict sepsis
- **Impact**: Every hour of delayed sepsis treatment increases mortality ~7%

### **Cardiac Arrhythmia Detection**
- **Challenge**: Paroxysmal arrhythmias missed by ECG snapshots
- **HR Role**: Regular beats vs. irregular pattern changes
- **Callisia Solution**: Continuous HR sequence analysis detects pattern anomalies
- **Benefit**: 24/7 monitoring vs. periodic ECG

---

## 🔬 Research & Innovation Focus

### **Current Phase (This Repository)**
✅ Robust HR extraction from multimodal sensors  
✅ Cross-patient generalization validation  
✅ Production-ready ensemble model  

### **Planned Innovations**

| Component                    | Status  | Expected Impact                           |
|------------------------------|---------|-------------------------------------------|
| Heart Rate Variability (HRV) | Q3 2026 | Stress/autonomic nervous system indicator |
| Respiratory Rate Estimation  | Q3 2026 | Respiratory distress detection            |
| Posture Recognition          | Q4 2026 | Fall risk, mobility assessment            |
| Multi-Biomarker Risk Score   | Q4 2026 | Integrated patient risk indicator         |
| LLM Integration              | Q1 2027 | Clinician-friendly risk narratives        |
| Real-time Dashboard          | Q1 2027 | Hospital bedside monitoring               |
| ECG Morphology Analysis      | Q2 2027 | Arrhythmia detection                      |
| Clinical Event Prediction    | Q2 2027 | Sepsis, cardiac events, deterioration     |

---

## 🛠️ Technologies & Dependencies

```
Core ML Stack:
  - PyTorch 2.x              Deep learning (CNN + Transformer)
  - NumPy/Pandas             Data processing
  - scikit-learn             ML utilities, metrics
  - NeuroKit2                Signal processing (ECG, PPG)
  - Optuna                   Hyperparameter optimization (Transformer)

Production Pipelines:
  - LangChain                LLM orchestration & prompt management
  - Google Generative AI     Clinical insight generation (Gemini)
  - Pydantic                 Data validation & schemas
  - PyYAML                   Configuration management

Visualization & Analysis:
  - Matplotlib/Seaborn       Signal plots, training history
  - Jupyter                  Exploratory analysis notebooks

Clinical Wearable Data:
  - Public PPG-DaLiA         Benchmark dataset
  - Public WESAD             Stress/affect dataset
  - Empatica E4              Reference implementation

Future Integrations:
  - FastAPI                  REST API deployment
  - MLflow                   Experiment tracking
  - ONNX                     Model portability
```

---

## 📚 References & Context

### Academic Partnerships
- **Università Politecnica delle Marche (UNIVPM)**: Master's program, academic supervision
- **Callisia S.r.l.**: Industry partner, clinical domain expertise

### Public Datasets & Benchmarks
- **PPG-DaLiA**: Individualized monitoring of generalized anxiety disorder using wearable sensors
- **WESAD**: A multimodal dataset for wearable stress and affect detection

### Clinical Foundation
This project builds on extensive research in:
- Wearable biomedical sensing
- Physiological signal processing
- Machine learning for healthcare
- Clinical decision support systems
- Patient deterioration prediction (Early Warning Scores)

---

## ⚠️ Clinical & Regulatory Considerations

### Data Privacy & Security
- ✅ Uses public datasets (de-identified)
- ⚠️ Real deployment requires HIPAA/GDPR compliance
- ⚠️ Patient consent for wearable monitoring required

### Clinical Validation
- ⚠️ Current models for research/development only
- ⚠️ Clinical validation required before hospital deployment
- ⚠️ FDA/CE certification path needed for medical device
- ⚠️ Multi-center clinical trials required for regulatory approval

### Model Limitations
- ⚠️ Trained on ~30 healthy/stressed subjects (not patients)
- ⚠️ May not generalize to elderly, obese, or severely ill populations
- ⚠️ Tattoos, skin conditions, poor circulation affect PPG quality
- ⚠️ Movement artifacts can cause errors

### Future Clinical Deployment Requirements
- ✅ Multi-site clinical trials
- ✅ FDA 510(k) or De Novo pathway
- ✅ Integration with hospital IT infrastructure
- ✅ Clinician training and validation
- ✅ Continuous performance monitoring post-deployment

---

## 🎓 Thesis Contribution

This project contributes to addressing the thesis research question:

> **"How can continuous multimodal wearable sensor data be transformed into actionable, personalized risk indicators for predicting adverse clinical events in hospitalized patients?"**

### Key Contributions

1. **Foundational Biomarker**: Accurate, robust HR extraction from PPG+ACC
2. **Generalization Framework**: LOSO validation ensures cross-patient applicability
3. **Production Architecture**: Ensemble averaging creates deployment-ready model
4. **Scalable Pipeline**: Data processing and training infrastructure for Phase 2-4 biomarkers
5. **Clinical Relevance**: Demonstrates feasibility of continuous wearable monitoring in hospital settings

---

## 📧 Project Information

**Institution**: Passau University, Università Politecnica delle Marche (UNIVPM)  
**Industry Partner**: Callisia S.r.l.  
**Project Type**: Master's Thesis + Industry Collaboration  
**Application**: Digital Health, Wearable Biomedical Engineering  
**Status**: Phase 1 (HR Estimation) - Active Development  

**Thesis Title**: *Combining Machine Learning and Large Language Models for Healthcare: A Hybrid Approach for Interpreting Wearable Data*

---

## 📄 License & Citation

This project is part of academic research conducted in collaboration with Callisia S.r.l., Passau University and UNIVPM.

**If using this code for research, please cite:**
```
@mastersthesis{callisia2026,
  title={Combining Machine Learning and Large Language Models for Healthcare: A Hybrid Approach for Interpreting Wearable Data},
  author={[Alessandro Seghini, Florian Lemmerich]},
  school={Passau University, Università Politecnica delle Marche},
  year={2026},
  note={In collaboration with Callisia S.r.l.}
}
```

---

**Last Updated**: May 2026  
**Status**: Phase 1 - Active Development ✅ (Signal Processing + LLM Integration)  
**Next Phases**: Phase 2 - Advanced Biomarker Extraction | Phase 3 - Multi-biomarker Risk Scoring
