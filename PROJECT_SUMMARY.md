# NSL-KDD Intrusion Detection System — Project Summary

## Overview
A Streamlit application for network intrusion detection using the NSL-KDD dataset. Built for resource-constrained environments typical of small enterprises, it offers real-time analysis, multi-model training, and clear evaluation visualizations.

## Project Structure
```
near/
  app.py                 # Main Streamlit app
  requirements.txt       # Python dependencies
  README.md              # User documentation
  PROJECT_SUMMARY.md     # This document
  run_app.bat            # Windows launcher
  test_app.py            # Basic validation script
  NSL-KDD/
    NSL_KDD_Train.csv    # Training dataset
    NSL_KDD_Test.csv     # Test dataset
```

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. Open: `http://localhost:8501`

## Implemented Features
- Data overview with attack distributions
- Model training: Random Forest, Decision Tree, SVM
- Real-time detection with interactive inputs
- Performance analysis with metrics and plots

## Technical Specs
- Dataset: NSL-KDD (41 features; binary target is_attack)
- Frontend: Streamlit
- ML: scikit-learn (with StandardScaler for SVM path)
- Visualization: Plotly
- Metrics timing: psutil for CPU/RAM where available

## Application Pages
- Data Overview — dataset statistics and charts
- Model Training — train models; view metrics and confusion matrix
- Real-time Detection — manual input and prediction
- Performance Analysis — compare models; view curves and resource metrics

## Model Evaluation & Visualizations
Reported per model:
- Accuracy, Precision, Recall, F1 (weighted and macro)
- ROC-AUC and PR-AUC
- Confusion matrices and classification reports
- ROC and Precision-Recall curves
- Latency (ms/sample), CPU time (ms/sample), RAM usage (MB), artifact size (KB)

Notes:
- AUC metrics use probabilities when available, else decision scores.
- Results vary with the train/test split and dataset sampling.

## Testing
Use `test_app.py` to quickly validate data loading, preprocessing, and a simple training pipeline.

## Future Enhancements
- Additional models (e.g., XGBoost, calibrated probabilistic SVM)
- Per-attack-type breakdowns and cost-sensitive metrics
- Model persistence and deployment profiles

## Support
If something breaks:
- Verify dataset files exist under `NSL-KDD/`
- Reinstall requirements
- Check app logs in the terminal

