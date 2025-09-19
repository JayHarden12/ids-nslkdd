# NSL-KDD Intrusion Detection System

A resource-efficient web application for network intrusion detection using the NSL-KDD dataset, tailored for small enterprises. The app provides real-time threat analysis, multi-model training, and rich performance visualizations.

## Features

### Core Functionality
- Real-time intrusion detection with interactive inputs
- Multiple ML models: Random Forest, Decision Tree, and SVM
- Streamlit-based UI with responsive, multi-page navigation
- Comprehensive analytics and visualizations

### Data Analysis
- Dataset overview with key statistics and distributions
- Attack type breakdowns (Normal vs Attack)
- Performance metrics: Accuracy, Precision, Recall, F1 (weighted and macro), ROC-AUC, PR-AUC
- Model diagnostics: confusion matrices, ROC and Precision-Recall curves

### Detection Capabilities
- Binary classification: Normal vs Attack
- Confidence scoring where available (probabilities or decision scores)
- Manual input form for real-time traffic analysis

## Installation

### Prerequisites
- Python 3.8+
- NSL-KDD dataset CSVs

### Setup
1. Ensure the following files exist:
   - `app.py`
   - `requirements.txt`
   - `NSL-KDD/NSL_KDD_Train.csv`
   - `NSL-KDD/NSL_KDD_Test.csv`
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   streamlit run app.py
   ```
4. Open `http://localhost:8501` in your browser.

## Usage

### Pages
- Data Overview: dataset statistics and distributions
- Model Training: train models and view detailed metrics
- Real-time Detection: analyze custom traffic inputs
- Performance Analysis: compare models and view diagnostic plots

### Real-time Detection
1. Go to "Real-time Detection"
2. Fill in the network parameters (e.g., protocol, service, bytes, counts)
3. Select a trained model
4. Click "Analyze Traffic" and review the prediction and confidence

### Dataset on Streamlit Cloud
- Preferred: Commit the real dataset files to the repo so they are available at runtime (no uploads needed).
- Accepted filenames and locations (case-sensitive on Linux):
  - `NSL-KDD/NSL_KDD_Train.csv` and `NSL-KDD/NSL_KDD_Test.csv`, or
  - `NSL-KDD/KDDTrain+.txt` and `NSL-KDD/KDDTest+.txt`
- The app auto-discovers these names (.csv or .txt). If files are elsewhere, place them under `NSL-KDD/`.
- Do not use Git LFS for these files on Streamlit Cloud — LFS pointers may be cloned without the actual data. Commit the actual CSV/TXT content, or add a download step.
- If the app still cannot find the data, you can optionally upload Train/Test or a combined CSV from the UI to proceed.

## Evaluation Metrics and Plots

Per model, the app reports:
- Accuracy, Precision, Recall, F1 (weighted and macro)
- ROC-AUC and PR-AUC
- Confusion matrix and classification report
- ROC and Precision-Recall curves
- Latency (ms/sample), CPU time (ms/sample), RAM usage (MB), and artifact size (KB)

Notes:
- AUC metrics use probability estimates when available; otherwise decision scores are used.
- Metrics vary with train/test split; refer to the app’s tables and charts for actual values.

## Dataset

- Training: 125,972 records; Test: 22,544 records
- 41 features across basic, content, traffic, and host-based categories
- Attack types grouped into Normal vs Attack for binary detection in the app

## Technology Stack
- Frontend: Streamlit
- ML: scikit-learn
- Data: pandas, NumPy
- Visualization: Plotly (plus Matplotlib/Seaborn as needed)

## Contributing
- Issues and PRs for models, visualizations, and performance improvements are welcome.

## License
- MIT License

## Support
- If you have issues, check dataset paths, install requirements, and review terminal logs. Then open an issue with details.
