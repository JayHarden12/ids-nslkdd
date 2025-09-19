import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import os
import pickle
import time
import io
import re
from pathlib import Path

try:
    import psutil
except Exception:
    psutil = None


st.set_page_config(
    page_title="NSL-KDD Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; color: #1f77b4; text-align: center; margin-bottom: 1.2rem; }
    .metric-card { background-color: #f0f2f6; padding: 0.75rem; border-radius: 0.5rem; border-left: 4px solid #1f77b4; }
    .prediction-box { background-color: #e8f4fd; padding: 0.75rem; border-radius: 0.5rem; border: 2px solid #1f77b4; margin: 0.5rem 0; }
    .attack-type { font-weight: 700; color: #d62728; }
    .normal-type { font-weight: 700; color: #2ca02c; }
</style>
""",
    unsafe_allow_html=True,
)


# NSL-KDD column names (standard)
FEATURE_NAMES = [
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


def _is_git_lfs_pointer(file_path: Path) -> bool:
    try:
        if not file_path.exists() or file_path.stat().st_size > 1024:
            return False
        text = file_path.read_text(errors='ignore')
        return 'git-lfs' in text and 'version https://git-lfs.github.com/spec/v1' in text
    except Exception:
        return False


def _discover_nsl_kdd_files(base_dir: Path) -> tuple[Path | None, Path | None]:
    """Attempt to find NSL-KDD train/test files with common names.
    Returns (train_path, test_path) or (None, None) if not found.
    """
    candidates_train = re.compile(r"^(NSL[_-]KDD[_-]?Train|KDDTrain\+).*\.(csv|txt)$", re.I)
    candidates_test = re.compile(r"^(NSL[_-]KDD[_-]?Test|KDDTest\+).*\.(csv|txt)$", re.I)

    train_path = None
    test_path = None

    search_roots = [base_dir / 'NSL-KDD', base_dir]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob('*'):
            if not p.is_file():
                continue
            name = p.name
            if train_path is None and candidates_train.match(name):
                train_path = p
            elif test_path is None and candidates_test.match(name):
                test_path = p
            if train_path is not None and test_path is not None:
                return train_path, test_path
    return train_path, test_path


def _read_nsl_kdd_dataframe(obj) -> pd.DataFrame:
    """Read NSL-KDD CSV/TXT from a path or file-like and normalize columns.
    - Handles presence/absence of header.
    - Drops any extra columns like 'difficulty' if present.
    """
    # Try reading with automatic delimiter detection
    df = pd.read_csv(obj, header=None)
    # If first row looks like header, re-read with header=0
    first_cell = str(df.iloc[0, 0]) if not df.empty else ''
    if first_cell.lower() == 'duration':
        df = pd.read_csv(obj, header=0)
    # Normalize to first 42 columns (41 features + attack_type)
    if df.shape[1] < 42:
        # Retry with engine='python' and sep=',' in case of parsing issues
        df = pd.read_csv(obj, header=None, engine='python', sep=',')
        first_cell = str(df.iloc[0, 0]) if not df.empty else ''
        if first_cell.lower() == 'duration':
            df = pd.read_csv(obj, header=0, engine='python', sep=',')
    # After retry, enforce at least 42 columns
    if df.shape[1] < 42:
        raise ValueError(f"Expected >=42 columns, found {df.shape[1]}")
    df = df.iloc[:, :42]
    df.columns = FEATURE_NAMES
    return df


@st.cache_data
def load_data(uploaded_train_bytes: bytes = None,
              uploaded_test_bytes: bytes = None,
              uploaded_combined_bytes: bytes = None):
    """Load NSL-KDD data from local repo or uploaded bytes.

    Priority:
    1) uploaded_combined_bytes (single CSV containing train+test)
    2) uploaded_train_bytes + uploaded_test_bytes
    3) Discover local files under repo (e.g., NSL-KDD/KDDTrain+.txt, NSL_KDD_Train.csv)
    """
    try:
        if uploaded_combined_bytes is not None:
            buf = io.BytesIO(uploaded_combined_bytes)
            df = _read_nsl_kdd_dataframe(buf)
        elif uploaded_train_bytes is not None and uploaded_test_bytes is not None:
            train_df = _read_nsl_kdd_dataframe(io.BytesIO(uploaded_train_bytes))
            test_df = _read_nsl_kdd_dataframe(io.BytesIO(uploaded_test_bytes))
            df = pd.concat([train_df, test_df], ignore_index=True)
        else:
            base = Path('.')
            train_path, test_path = _discover_nsl_kdd_files(base)
            if train_path is None or test_path is None:
                return None
            # Detect Git LFS pointers
            if _is_git_lfs_pointer(train_path) or _is_git_lfs_pointer(test_path):
                st.error("The dataset files appear to be Git LFS pointers. Streamlit Cloud may not pull LFS content. Replace them with actual CSVs or add a download step.")
                return None
            train_df = _read_nsl_kdd_dataframe(train_path)
            test_df = _read_nsl_kdd_dataframe(test_path)
            df = pd.concat([train_df, test_df], ignore_index=True)

        df = df.dropna()
        df['is_attack'] = df['attack_type'].apply(lambda x: 0 if str(x).strip().lower() == 'normal' else 1)
        return df
    except Exception:
        return None


def generate_synthetic_nsl_kdd(n_rows: int = 5000) -> pd.DataFrame:
    """Generate a small synthetic dataset with NSL-KDD-like columns.
    This enables the app to run when the real dataset isn't available.
    """
    rng = np.random.default_rng(42)
    protocols = ['tcp', 'udp', 'icmp']
    services = ['http', 'smtp', 'ftp', 'domain_u', 'auth', 'telnet', 'finger', 'pop_3']
    flags = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO']

    df = pd.DataFrame({
        'duration': rng.integers(0, 1000, size=n_rows),
        'protocol_type': rng.choice(protocols, size=n_rows),
        'service': rng.choice(services, size=n_rows),
        'flag': rng.choice(flags, size=n_rows),
        'src_bytes': rng.integers(0, 100000, size=n_rows),
        'dst_bytes': rng.integers(0, 100000, size=n_rows),
        'land': rng.integers(0, 2, size=n_rows),
        'wrong_fragment': rng.integers(0, 3, size=n_rows),
        'urgent': rng.integers(0, 3, size=n_rows),
        'hot': rng.integers(0, 5, size=n_rows),
        'num_failed_logins': rng.integers(0, 5, size=n_rows),
        'logged_in': rng.integers(0, 2, size=n_rows),
        'num_compromised': rng.integers(0, 5, size=n_rows),
        'root_shell': rng.integers(0, 2, size=n_rows),
        'su_attempted': rng.integers(0, 2, size=n_rows),
        'num_root': rng.integers(0, 10, size=n_rows),
        'num_file_creations': rng.integers(0, 5, size=n_rows),
        'num_shells': rng.integers(0, 3, size=n_rows),
        'num_access_files': rng.integers(0, 5, size=n_rows),
        'num_outbound_cmds': 0,
        'is_host_login': rng.integers(0, 2, size=n_rows),
        'is_guest_login': rng.integers(0, 2, size=n_rows),
        'count': rng.integers(0, 100, size=n_rows),
        'srv_count': rng.integers(0, 100, size=n_rows),
        'serror_rate': rng.random(size=n_rows),
        'srv_serror_rate': rng.random(size=n_rows),
        'rerror_rate': rng.random(size=n_rows),
        'srv_rerror_rate': rng.random(size=n_rows),
        'same_srv_rate': rng.random(size=n_rows),
        'diff_srv_rate': rng.random(size=n_rows),
        'srv_diff_host_rate': rng.random(size=n_rows),
        'dst_host_count': rng.integers(0, 255, size=n_rows),
        'dst_host_srv_count': rng.integers(0, 255, size=n_rows),
        'dst_host_same_srv_rate': rng.random(size=n_rows),
        'dst_host_diff_srv_rate': rng.random(size=n_rows),
        'dst_host_same_src_port_rate': rng.random(size=n_rows),
        'dst_host_srv_diff_host_rate': rng.random(size=n_rows),
        'dst_host_serror_rate': rng.random(size=n_rows),
        'dst_host_srv_serror_rate': rng.random(size=n_rows),
        'dst_host_rerror_rate': rng.random(size=n_rows),
        'dst_host_srv_rerror_rate': rng.random(size=n_rows),
    })
    # Create attack labels
    df['attack_type'] = rng.choice(['normal', 'neptune', 'smurf', 'satan', 'ipsweep'], size=n_rows,
                                   p=[0.8, 0.08, 0.06, 0.03, 0.03])
    return df


@st.cache_data
def preprocess_data(df: pd.DataFrame):
    le_protocol = LabelEncoder()
    le_service = LabelEncoder()
    le_flag = LabelEncoder()

    df = df.copy()
    df['protocol_type_encoded'] = le_protocol.fit_transform(df['protocol_type'])
    df['service_encoded'] = le_service.fit_transform(df['service'])
    df['flag_encoded'] = le_flag.fit_transform(df['flag'])

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
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'protocol_type_encoded',
        'service_encoded', 'flag_encoded'
    ]

    X = df[feature_columns]
    y = df['is_attack']
    return X, y, le_protocol, le_service, le_flag


@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42),
    }

    trained_models = {}
    results = {}

    for name, model in models.items():
        if name == 'SVM':
            model.fit(X_train_scaled, y_train)
            X_eval = X_test_scaled
        else:
            model.fit(X_train, y_train)
            X_eval = X_test

        y_pred = model.predict(X_eval)

        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X_eval)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_score = model.decision_function(X_eval)
        else:
            y_score = y_pred

        # Measure latency and CPU/RAM
        process = psutil.Process(os.getpid()) if psutil is not None else None
        start_cpu = process.cpu_times() if process else None
        t0 = time.perf_counter()
        _ = model.predict(X_eval)
        t1 = time.perf_counter()
        latency_ms_per_sample = (t1 - t0) * 1000.0 / len(X_eval)
        if process and start_cpu:
            end_cpu = process.cpu_times()
            cpu_delta = (end_cpu.user - start_cpu.user) + (end_cpu.system - start_cpu.system)
            cpu_ms_per_sample = (cpu_delta * 1000.0) / len(X_eval)
            ram_mb = process.memory_info().rss / (1024 * 1024)
        else:
            cpu_ms_per_sample = None
            ram_mb = None

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
        try:
            rocauc = roc_auc_score(y_test, y_score)
        except Exception:
            rocauc = None
        try:
            prauc = average_precision_score(y_test, y_score)
        except Exception:
            prauc = None

        # Curves
        try:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_score)
            curves = {'roc': {'fpr': fpr, 'tpr': tpr}, 'pr': {'precision': prec_curve, 'recall': rec_curve}}
        except Exception:
            curves = None

        try:
            artifact_kb = len(pickle.dumps(model)) / 1024.0
        except Exception:
            artifact_kb = None

        trained_models[name] = model
        results[name] = {
            'accuracy': accuracy,
            'precision': prec_w,
            'recall': rec_w,
            'f1': f1_w,
            'precision_macro': prec_m,
            'recall_macro': rec_m,
            'f1_macro': f1_m,
            'roc_auc': rocauc,
            'pr_auc': prauc,
            'curves': curves,
            'latency_ms_per_sample': latency_ms_per_sample,
            'cpu_ms_per_sample': cpu_ms_per_sample,
            'ram_mb': ram_mb,
            'artifact_kb': artifact_kb,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_score': y_score,
        }

    return trained_models, results, scaler, X_test, y_test


def create_visualizations(df, results):
    fig1 = px.pie(
        df,
        names='attack_type',
        title='Distribution of Attack Types',
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')

    fig2 = px.pie(
        df,
        names='is_attack',
        title='Normal vs Attack Distribution',
        color_discrete_map={0: '#2ca02c', 1: '#d62728'},
    )
    fig2.update_traces(textposition='inside', textinfo='percent+label')

    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names] if model_names else []

    if model_names:
        perf_df = pd.DataFrame({'Model': model_names, 'Accuracy': accuracies})
        fig3 = px.bar(
            perf_df,
            x='Model',
            y='Accuracy',
            title='Model Accuracy Comparison',
            color='Accuracy',
            color_continuous_scale='Viridis',
        )
        fig3.update_layout(showlegend=False)
    else:
        fig3 = go.Figure()
        fig3.update_layout(
            title='Model Accuracy Comparison', xaxis_title='Models', yaxis_title='Accuracy', xaxis=dict(visible=False)
        )
        fig3.add_annotation(
            text='Train models to view accuracy comparison', x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False
        )

    return fig1, fig2, fig3


def predict_attack(model, scaler, input_data, le_protocol, le_service, le_flag):
    try:
        input_data = input_data.copy()
        input_data['protocol_type_encoded'] = le_protocol.transform([input_data['protocol_type']])[0]
        input_data['service_encoded'] = le_service.transform([input_data['service']])[0]
        input_data['flag_encoded'] = le_flag.transform([input_data['flag']])[0]

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
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'protocol_type_encoded',
            'service_encoded', 'flag_encoded'
        ]

        X = np.array([input_data.get(col, 0) for col in feature_columns]).reshape(1, -1)

        if model.__class__.__name__ == 'SVC':
            X = scaler.transform(X)

        prediction = model.predict(X)[0]
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(X)[0]
        return prediction, probability
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None


def page_overview(df):
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Features", len(FEATURE_NAMES) - 1)
    with col3:
        st.metric("Normal Records", f"{len(df[df['is_attack'] == 0]):,}")
    with col4:
        st.metric("Attack Records", f"{len(df[df['is_attack'] == 1]):,}")

    fig1, fig2, fig3 = create_visualizations(df, {})
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fig1, use_container_width=True)
    with c2:
        st.plotly_chart(fig2, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Sample Data")
    st.dataframe(df.head(10))


def page_train(df):
    st.header("Machine Learning Model Training")
    with st.spinner("Preprocessing data and training models..."):
        X, y, le_protocol, le_service, le_flag = preprocess_data(df)
        trained_models, results, scaler, X_test, y_test = train_models(X, y)

    st.success("Models trained successfully!")

    metrics_rows = []
    for m, res in results.items():
        metrics_rows.append({
            'Model': m,
            'Accuracy': res['accuracy'],
            'Precision (Weighted)': res['precision'],
            'Recall (Weighted)': res['recall'],
            'F1-Weighted': res['f1'],
            'Precision (Macro)': res['precision_macro'],
            'Recall (Macro)': res['recall_macro'],
            'F1-Macro': res['f1_macro'],
            'ROC-AUC': res['roc_auc'],
            'PR-AUC': res['pr_auc'],
            'Latency (ms/sample)': res['latency_ms_per_sample'],
            'CPU (ms/sample)': res['cpu_ms_per_sample'],
            'RAM (MB)': res['ram_mb'],
            'Artifact Size (KB)': res['artifact_kb'],
        })
    results_df = pd.DataFrame(metrics_rows)
    st.dataframe(results_df.round(4))

    best_model_name = results_df.sort_values('Accuracy', ascending=False)['Model'].iloc[0]
    st.success(f"Best performing model: {best_model_name} (Accuracy {results[best_model_name]['accuracy']:.4f})")

    st.subheader("Confusion Matrix")
    model_for_cm = st.selectbox("Select model", list(results.keys()), key='train_cm_model')
    cm = confusion_matrix(results[model_for_cm]['y_test'], results[model_for_cm]['y_pred'])
    fig_cm = px.imshow(cm, text_auto=True, labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix - {model_for_cm}")
    fig_cm.update_xaxes(side="bottom")
    st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report")
    st.text(classification_report(results[model_for_cm]['y_test'], results[model_for_cm]['y_pred'], target_names=['Normal', 'Attack']))


def page_realtime(df):
    st.header("Real-time Intrusion Detection")
    with st.spinner("Loading trained models..."):
        X, y, le_protocol, le_service, le_flag = preprocess_data(df)
        trained_models, results, scaler, X_test, y_test = train_models(X, y)

    st.subheader("Input Network Traffic Parameters")
    col1, col2 = st.columns(2)
    with col1:
        duration = st.number_input("Duration", min_value=0.0, value=0.0, step=0.1)
        protocol_type = st.selectbox("Protocol Type", sorted(df['protocol_type'].unique().tolist()))
        service = st.selectbox("Service", sorted(df['service'].unique().tolist())[:50])
        flag = st.selectbox("Flag", sorted(df['flag'].unique().tolist()))
        src_bytes = st.number_input("Source Bytes", min_value=0, value=0)
        dst_bytes = st.number_input("Destination Bytes", min_value=0, value=0)
        logged_in = st.selectbox("Logged In", [0, 1])
    with col2:
        land = st.selectbox("Land", [0, 1])
        wrong_fragment = st.number_input("Wrong Fragment", min_value=0, value=0)
        urgent = st.number_input("Urgent", min_value=0, value=0)
        hot = st.number_input("Hot", min_value=0, value=0)
        num_failed_logins = st.number_input("Failed Logins", min_value=0, value=0)
        count = st.number_input("Count", min_value=0, value=0)
        srv_count = st.number_input("Service Count", min_value=0, value=0)

    input_data = {
        'duration': duration,
        'protocol_type': protocol_type,
        'service': service,
        'flag': flag,
        'src_bytes': src_bytes,
        'dst_bytes': dst_bytes,
        'land': land,
        'wrong_fragment': wrong_fragment,
        'urgent': urgent,
        'hot': hot,
        'num_failed_logins': num_failed_logins,
        'logged_in': logged_in,
        'num_compromised': 0,
        'root_shell': 0,
        'su_attempted': 0,
        'num_root': 0,
        'num_file_creations': 0,
        'num_shells': 0,
        'num_access_files': 0,
        'num_outbound_cmds': 0,
        'is_host_login': 0,
        'is_guest_login': 0,
        'count': count,
        'srv_count': srv_count,
        'serror_rate': 0.0,
        'srv_serror_rate': 0.0,
        'rerror_rate': 0.0,
        'srv_rerror_rate': 0.0,
        'same_srv_rate': 0.0,
        'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 0,
        'dst_host_srv_count': 0,
        'dst_host_same_srv_rate': 0.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0,
    }

    selected_model = st.selectbox("Select Model for Prediction", list(trained_models.keys()))
    if st.button("Analyze Traffic", type="primary"):
        with st.spinner("Analyzing network traffic..."):
            prediction, probability = predict_attack(
                trained_models[selected_model], scaler, input_data, le_protocol, le_service, le_flag
            )
        if prediction is not None:
            col1, col2 = st.columns(2)
            with col1:
                if prediction == 0:
                    st.markdown('<div class="prediction-box"><p class="normal-type">NORMAL TRAFFIC</p></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="prediction-box"><p class="attack-type">POTENTIAL ATTACK DETECTED</p></div>', unsafe_allow_html=True)
            with col2:
                if probability is not None:
                    st.metric("Normal Probability", f"{probability[0]*100:.2f}%")
                    st.metric("Attack Probability", f"{probability[1]*100:.2f}%")
                else:
                    st.info("Probabilities not available for this model configuration.")


def page_performance(df):
    st.header("Model Performance Analysis")
    with st.spinner("Analyzing model performance..."):
        X, y, le_protocol, le_service, le_flag = preprocess_data(df)
        trained_models, results, scaler, X_test, y_test = train_models(X, y)

    st.subheader("Model Performance Comparison")
    metrics_data = []
    for model_name, result in results.items():
        metrics_data.append({
            'Model': model_name,
            'Accuracy': result.get('accuracy'),
            'Precision (Weighted)': result.get('precision'),
            'Recall (Weighted)': result.get('recall'),
            'F1-Weighted': result.get('f1'),
            'Precision (Macro)': result.get('precision_macro'),
            'Recall (Macro)': result.get('recall_macro'),
            'F1-Macro': result.get('f1_macro'),
            'ROC-AUC': result.get('roc_auc'),
            'PR-AUC': result.get('pr_auc'),
            'Latency (ms/sample)': result.get('latency_ms_per_sample'),
            'CPU (ms/sample)': result.get('cpu_ms_per_sample'),
            'RAM (MB)': result.get('ram_mb'),
            'Artifact Size (KB)': result.get('artifact_kb'),
        })
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df.round(4))

    col1, col2 = st.columns(2)
    with col1:
        fig_acc = px.bar(metrics_df, x='Model', y='Accuracy', title='Model Accuracy', color='Accuracy', color_continuous_scale='Viridis')
        fig_acc.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_acc, use_container_width=True)
    with col2:
        fig_f1w = px.bar(metrics_df, x='Model', y='F1-Weighted', title='F1 (Weighted)', color='F1-Weighted', color_continuous_scale='Plasma')
        fig_f1w.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_f1w, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig_prec = px.bar(metrics_df, x='Model', y='Precision (Weighted)', title='Precision (Weighted)', color='Precision (Weighted)', color_continuous_scale='Cividis')
        fig_prec.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_prec, use_container_width=True)
    with col4:
        fig_rec = px.bar(metrics_df, x='Model', y='Recall (Weighted)', title='Recall (Weighted)', color='Recall (Weighted)', color_continuous_scale='Magma')
        fig_rec.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_rec, use_container_width=True)

    col5, col6 = st.columns(2)
    with col5:
        fig_f1m = px.bar(metrics_df, x='Model', y='F1-Macro', title='F1 (Macro)', color='F1-Macro', color_continuous_scale='Bluered')
        fig_f1m.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_f1m, use_container_width=True)
    with col6:
        fig_rocauc = px.bar(metrics_df, x='Model', y='ROC-AUC', title='ROC-AUC', color='ROC-AUC', color_continuous_scale='Greens')
        fig_rocauc.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_rocauc, use_container_width=True)

    col7, col8 = st.columns(2)
    with col7:
        fig_prauc = px.bar(metrics_df, x='Model', y='PR-AUC', title='PR-AUC', color='PR-AUC', color_continuous_scale='Oranges')
        fig_prauc.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_prauc, use_container_width=True)
    with col8:
        fig_latency = px.bar(metrics_df, x='Model', y='Latency (ms/sample)', title='Latency (ms/sample)', color='Latency (ms/sample)', color_continuous_scale='Turbo')
        fig_latency.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_latency, use_container_width=True)

    col9, col10 = st.columns(2)
    with col9:
        fig_cpu = px.bar(metrics_df, x='Model', y='CPU (ms/sample)', title='CPU Time (ms/sample)', color='CPU (ms/sample)', color_continuous_scale='Purples')
        fig_cpu.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_cpu, use_container_width=True)
    with col10:
        fig_art = px.bar(metrics_df, x='Model', y='Artifact Size (KB)', title='Artifact Size (KB)', color='Artifact Size (KB)', color_continuous_scale='Teal')
        fig_art.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_art, use_container_width=True)

    st.subheader("Comprehensive Model Comparison (Radar)")
    radar = go.Figure()
    for _, row in metrics_df.iterrows():
        radar.add_trace(
            go.Scatterpolar(
                r=[row['Accuracy'], row['Precision (Weighted)'], row['Recall (Weighted)'], row['F1-Weighted']],
                theta=['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Weighted'],
                fill='toself',
                name=row['Model'],
            )
        )
    radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=True)
    st.plotly_chart(radar, use_container_width=True)

    st.subheader("Per-Model Confusion Matrix and Curves")
    model_choice = st.selectbox("Select model", list(results.keys()), key='perf_model_select')
    sel = results[model_choice]
    cm_sel = confusion_matrix(sel['y_test'], sel['y_pred'])
    fig_cm_sel = px.imshow(cm_sel, text_auto=True, labels=dict(x="Predicted", y="Actual"), title=f"Confusion Matrix - {model_choice}")
    fig_cm_sel.update_xaxes(side="bottom")
    st.plotly_chart(fig_cm_sel, use_container_width=True)

    curves = sel.get('curves')
    if curves is not None:
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=curves['roc']['fpr'], y=curves['roc']['tpr'], mode='lines', name=model_choice))
        roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash')))
        roc_fig.update_layout(title=f"ROC Curve - {model_choice}", xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
        st.plotly_chart(roc_fig, use_container_width=True)

        pr_fig = go.Figure()
        pr_fig.add_trace(go.Scatter(x=curves['pr']['recall'], y=curves['pr']['precision'], mode='lines', name=model_choice))
        pr_fig.update_layout(title=f"Precision-Recall Curve - {model_choice}", xaxis_title='Recall', yaxis_title='Precision')
        st.plotly_chart(pr_fig, use_container_width=True)


def main():
    st.markdown('<h1 class="main-header">NSL-KDD Intrusion Detection System</h1>', unsafe_allow_html=True)
    st.markdown("### Resource-Efficient Machine Learning for Small Nigerian Enterprises")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Data Overview", "Model Training", "Real-time Detection", "Performance Analysis"],
    )

    # Try to load local dataset first
    with st.spinner("Loading NSL-KDD dataset..."):
        df = load_data()

    # If not found, provide upload UI and sample fallback
    if df is None:
        st.error("Failed to locate NSL-KDD dataset in the repo. Ensure the CSV/TXT files are committed (not Git LFS pointers) under `NSL-KDD/` with names like `NSL_KDD_Train.csv` and `NSL_KDD_Test.csv` or the standard `KDDTrain+.txt` and `KDDTest+.txt`.")
        st.caption("Optionally, you can upload the files below to proceed without changing the repo.")
        up_col1, up_col2 = st.columns(2)
        with up_col1:
            uploaded_train = st.file_uploader("Upload Train CSV/TXT (e.g., KDDTrain+.txt)", type=['csv','txt'], key='train_csv')
        with up_col2:
            uploaded_test = st.file_uploader("Upload Test CSV/TXT (e.g., KDDTest+.txt)", type=['csv','txt'], key='test_csv')
        uploaded_combined = st.file_uploader("Or upload a single combined CSV", type=['csv'], key='combined_csv')
        if uploaded_combined is not None:
            df = load_data(uploaded_combined_bytes=uploaded_combined.getvalue())
        elif uploaded_train is not None and uploaded_test is not None:
            df = load_data(uploaded_train_bytes=uploaded_train.getvalue(), uploaded_test_bytes=uploaded_test.getvalue())
        if df is None:
            st.stop()

    if page == "Data Overview":
        page_overview(df)
    elif page == "Model Training":
        page_train(df)
    elif page == "Real-time Detection":
        page_realtime(df)
    else:
        page_performance(df)


if __name__ == "__main__":
    main()
