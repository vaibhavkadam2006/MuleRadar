# MuleRadar

> An end-to-end Machine Learning pipeline and analytics platform designed to detect anomalous behavior, extract behavioral "DNA," and generate actionable alerts. 

---

## 📑 Project Overview

**MuleRadar** is a modular, five-phase machine learning system. While the specific domain (e.g., financial money mule detection, bot tracking, or network anomaly detection) depends on the ingested dataset, the architecture provides a complete lifecycle from raw data processing to real-time alerting and visualization. 

---

## 🏗️ System Architecture

The project is structured into modular phases, allowing for easy debugging, scaling, and independent execution.

| Phase | Component | File | Description |
| :--- | :--- | :--- | :--- |
| **Phase 1** | **Data Pipeline** | `src/phase1_data_pipeline.py` | Handles data ingestion, cleaning, normalization, and preprocessing of raw data streams. |
| **Phase 2** | **Feature DNA** | `src/phase2_feature_dna.py` | Performs advanced feature engineering, creating behavioral profiles or "DNA" signatures for entities. |
| **Phase 3** | **Model Training** | `src/phase3_model_training.py` | Trains predictive models/anomaly detectors using the engineered feature DNA. |
| **Phase 4** | **Alert Engine** | `src/phase4_alert_engine.py` | Runs inference on new data against the trained models, generating risk scores and triggering alerts. |
| **Phase 5** | **Dashboard** | `src/phase5_dashboard.py` | Provides a visual interface (likely via Streamlit or Dash) to monitor alerts, view model metrics, and analyze entity DNA. |
| **Interface** | **REST API** | `src/api.py` | Exposes endpoints for external applications to submit data for scoring or retrieve alerts. |

---

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* `pip` package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd MuleRadar
