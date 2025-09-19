#!/usr/bin/env python3
"""
Test script for the NSL-KDD Intrusion Detection System
This script tests the core functionality without running the full Streamlit app
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import os

def test_data_loading():
    """Test if the dataset can be loaded correctly"""
    print("Testing data loading...")
    
    try:
        # Load a small sample of the data
        df = pd.read_csv('NSL-KDD/NSL_KDD_Train.csv', nrows=1000)
        print(f"✅ Successfully loaded {len(df)} records")
        print(f"✅ Dataset shape: {df.shape}")
        return True
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_preprocessing():
    """Test data preprocessing functionality"""
    print("\nTesting data preprocessing...")
    
    try:
        # Load small sample
        df = pd.read_csv('NSL-KDD/NSL_KDD_Train.csv', nrows=1000)
        
        # Add column names
        feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
        ]
        
        df.columns = feature_names
        
        # Create binary classification
        df['is_attack'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Encode categorical variables
        le_protocol = LabelEncoder()
        le_service = LabelEncoder()
        le_flag = LabelEncoder()
        
        df['protocol_type_encoded'] = le_protocol.fit_transform(df['protocol_type'])
        df['service_encoded'] = le_service.fit_transform(df['service'])
        df['flag_encoded'] = le_flag.fit_transform(df['flag'])
        
        print("✅ Data preprocessing completed successfully")
        print(f"✅ Attack distribution: {df['is_attack'].value_counts().to_dict()}")
        return True
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return False

def test_model_training():
    """Test model training functionality"""
    print("\nTesting model training...")
    
    try:
        # Load and preprocess data
        df = pd.read_csv('NSL-KDD/NSL_KDD_Train.csv', nrows=1000)
        
        feature_names = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
            'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type'
        ]
        
        df.columns = feature_names
        df['is_attack'] = df['attack_type'].apply(lambda x: 0 if x == 'normal' else 1)
        
        # Select features
        feature_columns = [
            'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
            'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate'
        ]
        
        X = df[feature_columns]
        y = df['is_attack']
        
        # Train a simple model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Model training completed successfully")
        print(f"✅ Test accuracy: {accuracy:.4f}")
        return True
        
    except Exception as e:
        print(f"❌ Error in model training: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing NSL-KDD Intrusion Detection System")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists('NSL-KDD/NSL_KDD_Train.csv'):
        print("❌ Dataset not found. Please ensure NSL-KDD files are in the correct location.")
        return False
    
    tests = [
        test_data_loading,
        test_preprocessing,
        test_model_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The application should work correctly.")
        print("\nTo run the application, use:")
        print("streamlit run app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
