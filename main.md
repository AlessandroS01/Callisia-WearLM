## 🧬 Architecture Evolution & The Hybrid LLM Pipeline

This section documents the architectural evolution of the predictive model, detailing the transition from custom Deep Learning feature extractors to a State-of-the-Art (SOTA) foundational model, culminating in a clinical LLM decision-support agent.

### Phase 1: The CNN + BiLSTM Baseline
Initial approaches utilized a 1D-Convolutional Neural Network paired with a Bidirectional LSTM (up to 212K parameters). 
* **The Goal:** The CNN acted as a spatial feature extractor, while the BiLSTM provided temporal anchoring to prevent "amnesia" during sudden motion artifacts.
* **The Limitation:** The model exhibited a "Fear of Heights," systematically under-predicting high heart rates (HR) during intense physical activity. 
* **Cost-Sensitive Experiment:** A penalty weight (1.5x - 3.0x) was applied to the Huber/SmoothL1 loss for targets > 120 BPM. While this forced the model to predict higher rates, it induced mathematical "hallucinations" (over-corrections) during noisy accelerometer spikes, proving that standard CNNs fundamentally blur high-frequency physiological features.

### Phase 2: The `PatchHRNet` Exploration (Vision Transformer)
To overcome the CNN blurring effect, a custom Patch-Based Transformer (`PatchHRNet`) was engineered from scratch.
* **Signal Tokenization:** Instead of deep convolutions, an un-overlapped `Conv1d` layer tokenized 8-second waveform windows into 32 discrete, high-resolution patches (0.25s each), preserving the razor-thin physics of the Blood Volume Pulse (BVP).
* **Simultaneous Noise Rejection:** Utilizing `TransformerEncoderLayers`, the model evaluated all 32 patches simultaneously. If accelerometer noise corrupted patches 1-15, the attention matrix mathematically ignored them (multiplying influence by 0.0), extracting the HR strictly from the surviving clean patches.
* **Advanced Training Strategy:** To prevent attention collapse and overfitting on the DaLiA micro-dataset, a highly specialized training loop was developed:
  * **Unified Scheduler:** A single `LambdaLR` scheduler applying a Linear Warmup (first 5 epochs) followed by Cosine Annealing.
  * **GPU-Native Patch Masking:** A time-series equivalent of DropPath/Dropout that randomly zeroes out 20% of the sequence patches directly on the GPU, forcing the model to learn robust representations mimicking physical sensor failure.
  * **Optimization:** `AdamW` optimizer with heavy weight decay and strictly enforced Gradient Clipping (`max_norm=1.0`).

### Phase 3: The Pragmatic SOTA Pivot (BeliefPPG)
While `PatchHRNet` successfully demonstrated the superiority of Transformer architectures for this domain, training foundational time-series Transformers from scratch requires immense data volume and compute resources. 

To achieve the primary objective of this project—building an end-to-end clinical pipeline—the architecture was pragmatically pivoted. We integrated **BeliefPPG**, a publicly available, pre-trained SOTA model, to handle the raw feature extraction. This allowed us to successfully isolate the physical sensor noise and dedicate system resources to the downstream clinical logic.

### Phase 4: The LLM Tabular Handoff (Clinical Translation)
Raw heart rate integers provide limited value to medical professionals without context. The final stage of this pipeline bridges the gap between hardware and human diagnostics.

1. **Temporal Context Injection:** The data loader and inference engines were re-wired to capture and propagate ISO-8601 timestamps (the 0.0s mark of every 8-second window) alongside the ML predictions.
2. **Tabular Prompt Engineering:** The ML outputs are aggregated into a highly structured markdown table, combining the predicted HR, the exact timestamp, and patient metadata (e.g., Age, existing conditions).
3. **The Diagnostic Agent:** This tabular data is fed into a Large Language Model via API. The LLM acts as an intelligent safety net, utilizing clinical logic to assess the context of the HR trends (e.g., recognizing a spike to 140 BPM at 9:00 AM as a morning run, rather than a clinical anomaly requiring intervention).

### 🧪 Integration Testing Strategy
To safely test the plumbing of this hybrid pipeline without executing massive compute workloads, a dedicated dry-run methodology is utilized:
* **Test Branching:** Plumbing updates (like timestamp propagation) are isolated on branches (e.g., `test/end-to-end-pipeline`).
* **Model Mocking:** The PyTorch models are bypassed using hardcoded mock prediction tensors and simulated timestamps.
* **Validation:** This ensures the data cleanly flows from the raw CSVs, formats correctly into the LLM prompt, and successfully receives an API response before deploying full training loops.