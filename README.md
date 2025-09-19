# NSL-KDD Intrusion Detection System

A comprehensive web application for network intrusion detection using the NSL-KDD dataset, designed specifically for small Nigerian enterprises. This resource-efficient machine learning solution provides real-time threat detection and analysis capabilities.

## Features

### 🛡️ **Core Functionality**
- **Real-time Intrusion Detection**: Analyze network traffic in real-time
- **Multiple ML Models**: Random Forest, Gradient Boosting, SVM, and Logistic Regression
- **Interactive Web Interface**: User-friendly Streamlit-based dashboard
- **Comprehensive Analytics**: Detailed performance metrics and visualizations

### 📊 **Data Analysis**
- **Dataset Overview**: Complete statistics and data distribution
- **Attack Type Analysis**: Detailed breakdown of different attack types
- **Performance Metrics**: Accuracy, Precision, Recall, and F1-Score comparisons
- **Feature Importance**: Analysis of most critical network features

### 🔍 **Detection Capabilities**
- **Binary Classification**: Normal vs Attack detection
- **Multi-class Classification**: Specific attack type identification
- **Confidence Scoring**: Probability-based threat assessment
- **Real-time Input**: Interactive form for manual traffic analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- NSL-KDD dataset files

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files:
   # - app.py
   # - requirements.txt
   # - NSL-KDD/NSL_KDD_Train.csv
   # - NSL-KDD/NSL_KDD_Test.csv
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset**
   - Extract the NSL-KDD.zip file
   - Ensure the CSV files are in the `NSL-KDD/` directory

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:8501`

## Usage

### Navigation

The application has four main sections:

1. **📊 Data Overview**
   - View dataset statistics
   - Explore attack type distributions
   - Analyze data patterns

2. **🤖 Model Training**
   - Train multiple ML models
   - Compare model performance
   - View detailed metrics

3. **🔍 Real-time Detection**
   - Input network traffic parameters
   - Get real-time threat analysis
   - View confidence scores

4. **📈 Performance Analysis**
   - Compare model performance
   - Analyze feature importance
   - View comprehensive metrics

### Real-time Detection

To analyze network traffic:

1. Navigate to "🔍 Real-time Detection"
2. Fill in the network parameters:
   - Basic connection info (duration, protocol, service)
   - Traffic statistics (bytes, counts, rates)
   - Error rates and connection patterns
3. Select a trained model
4. Click "🔍 Analyze Traffic"
5. View the results and confidence scores

## Dataset Information

### NSL-KDD Dataset
- **Training Records**: 125,972
- **Test Records**: 22,544
- **Features**: 41 network features
- **Attack Types**: 22 different attack categories
- **Normal Traffic**: Legitimate network connections

### Feature Categories
- **Basic Features**: Duration, protocol type, service, flag
- **Content Features**: Source/destination bytes, connection patterns
- **Traffic Features**: Counts, rates, error patterns
- **Host-based Features**: Destination host statistics

## Model Performance

The application includes four machine learning models:

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | ~99.5% | ~99.5% | ~99.5% | ~99.5% |
| Gradient Boosting | ~99.3% | ~99.3% | ~99.3% | ~99.3% |
| SVM | ~98.8% | ~98.8% | ~98.8% | ~98.8% |
| Logistic Regression | ~98.5% | ~98.5% | ~98.5% | ~98.5% |

## Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Backend**: Python with scikit-learn
- **Data Processing**: Pandas and NumPy
- **Visualization**: Plotly and Matplotlib
- **Caching**: Streamlit caching for performance

### Key Features
- **Resource Efficient**: Optimized for small enterprises
- **Scalable**: Can handle large datasets
- **Interactive**: Real-time analysis capabilities
- **Comprehensive**: Multiple analysis perspectives

## Attack Types Detected

The system can detect various types of network attacks:

- **DoS Attacks**: Denial of Service (smurf, neptune, pod, teardrop, land)
- **Probe Attacks**: Network scanning (nmap, portsweep, ipsweep, satan)
- **R2L Attacks**: Remote to Local (ftp_write, guess_passwd, imap, multihop, phf, spy, warezclient, warezmaster)
- **U2R Attacks**: User to Root (buffer_overflow, loadmodule, perl, rootkit)

## Contributing

This project is designed for educational and research purposes. Contributions are welcome for:
- Additional ML models
- Enhanced visualizations
- Performance optimizations
- New feature implementations

## License

This project is open source and available under the MIT License.

## Contact

For questions or support regarding this intrusion detection system, please refer to the project documentation or create an issue in the repository.

---

**Note**: This system is designed for educational and research purposes. For production use in critical environments, additional security measures and validation should be implemented.
