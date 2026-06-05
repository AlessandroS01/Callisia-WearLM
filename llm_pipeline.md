## 🧠 Clinical Telemetry AI Pipeline

### Overview
This project implements an enterprise-grade AI pipeline that translates raw physiological telemetry (Heart Rate via BVP, Movement via 3-axis Accelerometer) from the LEAF device into structured, medical-grade clinical reports. It utilizes an **Orchestrator Pattern** to completely decouple the mathematical signal processing from the LLM reasoning engine and the final UI/Document generation layer.

### System Architecture
The backend is structured into distinct, modular pipelines managed by the `PipelineOrchestrator`:

1. **SignalProcessingPipeline:** Ingests raw Bluetooth sensor data and infers raw physiological metrics.
2. **ClinicalAggregator:** Groups the raw arrays into a statistical payload (means, medians, correlation scores).
3. **LLMInsightsPipeline:** Feeds the mathematical context into Google's Gemini model to generate human-readable clinical insights.
4. **ClinicalReportGeneratorPipeline:** Takes the validated AI output and renders an HTML-enhanced Markdown document (or PDF) for clinical triage.

### The Pydantic Schema (`ClinicalReportOutput`)
To prevent LLM hallucinations and ensure structural accuracy, the AI's output is strictly coerced into a Pydantic V2 schema. This guarantees the downstream UI always receives valid data types.

Key features of the schema include:
* **Chain of Thought (`internal_reasoning`):** Forces the LLM to deduce the math *before* formulating an observation. Hidden from the final UI.
* **Clinical Bounds:** Separates findings into `cardiovascular_state`, `autonomic_tone`, and `movement_context`.
* **Standardized Scoring (`ews_hr_score`):** Deterministically calculates a modified National Early Warning Score (NEWS2) based strictly on the heart rate thresholds.
* **Boolean Triage (`requires_attention`):** A strict `True/False` flag triggered by the EWS score or detected anomalies to drive red/green UI alerts.
* **Operational Directives (`recommended_system_action`):** Suggests next steps for the technician (e.g., verifying sensor calibration).

### Prompt Engineering & AI Safety Guardrails
The system employs a heavily constrained `CLINICAL_SYSTEM_PROMPT` designed to meet healthcare safety standards:
* **The "No Diagnosis" Rule:** The LLM is sandboxed into an "observer" role and explicitly forbidden from diagnosing medical conditions (e.g., it can state "high HR volatility", but cannot diagnose "Atrial Fibrillation").
* **Data Authority Override:** The LLM is explicitly instructed to trust the deterministic math layer over its own pre-training bias (e.g., if the math labels a 0.35 correlation as "Moderate", the LLM cannot override it to "Low").
* **Pre-Bunking Artifacts:** The prompt informs the LLM of expected ML architectural artifacts (like array length mismatches due to model warm-up) so it does not flag them as sensor errors.

### Resilience and Error Handling
Because LLMs are reliant on external APIs, the system is designed to be indestructible against network volatility:
* **Exponential Backoff:** API calls are wrapped with retry logic (e.g., `tenacity` / LangChain `.with_retry()`). If the Google servers return an `HTTP 503: Service Unavailable` due to high global traffic, the daemon automatically pauses and retries, rendering network spikes invisible to the user.
* **Encoding Safety:** File generation enforces `UTF-8` encoding to prevent legacy Windows `cp1252` crashes when rendering clinical emojis (🚨, ✅).

### UI and Report Generation
The system decouples the AI from the UI. The LLM only generates JSON. The `ClinicalReportGeneratorPipeline` injects this JSON into an HTML-enhanced Markdown template using flush-left `f-strings`. 

This approach forces Markdown renderers to constrain text widths using HTML tables, creating a highly scannable, dashboard-style medical chart complete with collapsible technical notes and isolated alert banners.

### Testing Strategy
Accuracy is verified across three domains:
1. **Structural:** Pydantic validation ensures the JSON never breaks the frontend.
2. **Semantic (Golden Datasets):** Parameterized `pytest` files run edge-case payloads (e.g., "Resting Tachycardia") against the pipeline to ensure the boolean `requires_attention` flags trigger correctly.
3. **LangSmith Tracing:** Integrated for "LLM-as-a-Judge" evaluation, allowing continuous monitoring of the semantic quality and tone of the generated reports.