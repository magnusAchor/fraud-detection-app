# ============================================================
# app.py — Main Streamlit Application
# Credit Card Fraud Detection Demo
# ============================================================

import streamlit as st
import numpy as np
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TF warnings

# ── Page config (must be first Streamlit call) ──────────────
st.set_page_config(
    page_title = "Fraud Detection AI",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ── Load config and model ────────────────────────────────────
@st.cache_resource
def load_model_and_config():
    """Rebuild model from params + weights — version independent."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model

    # Load params
    with open('model_params.json') as f:
        p = json.load(f)

    # Rebuild architecture exactly as trained
    inputs  = keras.Input(shape=(p['input_dim'],))
    x       = layers.Dense(p['enc1'],       activation='relu')(inputs)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(p['dropout_rate'])(x)
    x       = layers.Dense(p['enc2'],       activation='relu')(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dropout(p['dropout_rate'])(x)
    encoded = layers.Dense(p['bottleneck'], activation='relu')(x)
    x       = layers.Dense(p['enc2'],       activation='relu')(encoded)
    x       = layers.BatchNormalization()(x)
    x       = layers.Dense(p['enc1'],       activation='relu')(x)
    x       = layers.BatchNormalization()(x)
    outputs = layers.Dense(p['input_dim'],  activation='linear')(x)

    model   = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')

    # Load saved weights into the rebuilt architecture
    model.load_weights('autoencoder_weights.weights.h5')

    with open('app_config.json') as f:
        config = json.load(f)

    return model, config

model, config = load_model_and_config()

THRESHOLD     = config['threshold']
FEATURE_NAMES = config['feature_names']
INPUT_DIM     = config['input_dim']
RESULTS       = config['model_results']

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .fraud-box {
        background: linear-gradient(135deg, #ff4757, #c0392b);
        padding: 20px; border-radius: 12px; text-align: center;
        color: white; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(255,71,87,0.4);
    }
    .normal-box {
        background: linear-gradient(135deg, #2ed573, #27ae60);
        padding: 20px; border-radius: 12px; text-align: center;
        color: white; font-size: 28px; font-weight: bold;
        box-shadow: 0 4px 15px rgba(46,213,115,0.4);
    }
    .metric-card {
        background: #f8f9fa; padding: 15px;
        border-radius: 10px; text-align: center;
        border-left: 4px solid #3498db;
    }
    .stProgress .st-bo { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)


# ── Helper functions ─────────────────────────────────────────
def predict_transaction(features: np.ndarray):
    """Run a transaction through the autoencoder and return results."""
    features_2d    = features.reshape(1, -1).astype(np.float32)
    reconstructed  = model.predict(features_2d, verbose=0)
    error          = float(np.mean(np.power(features_2d - reconstructed, 2)))
    is_fraud       = error > THRESHOLD

    # Confidence: how far is the error from the threshold?
    normal_err     = config['normal_mean_error']
    fraud_err      = config['fraud_mean_error']
    if is_fraud:
        confidence = min(100, int((error - THRESHOLD) /
                         (fraud_err - THRESHOLD) * 100 + 50))
    else:
        confidence = min(100, int((THRESHOLD - error) /
                         (THRESHOLD - normal_err) * 100 + 50))

    # Per-feature reconstruction error
    features_2d_r  = features.reshape(1, -1).astype(np.float32)
    feat_errors    = np.power(features_2d_r - reconstructed, 2).flatten()

    return {
        'is_fraud'       : is_fraud,
        'error'          : error,
        'threshold'      : THRESHOLD,
        'confidence'     : min(confidence, 99),
        'feature_errors' : feat_errors,
        'top_features'   : np.argsort(feat_errors)[::-1][:8]
    }


def generate_random_transaction(fraud: bool = False):
    """Generate a sample transaction for demo purposes."""
    np.random.seed(np.random.randint(0, 9999))
    if fraud:
        # Exaggerate some V-features known to correlate with fraud
        features = np.random.randn(INPUT_DIM) * 1.5
        features[0]  = np.random.uniform(-5, -2)   # V1 negative
        features[3]  = np.random.uniform(2, 6)      # V4 positive
        features[10] = np.random.uniform(-4, -1)    # V11 negative
        features[13] = np.random.uniform(-8, -3)    # V14 very negative
        features[-2] = np.random.uniform(0.5, 3)    # Amount_scaled high
    else:
        features = np.random.randn(INPUT_DIM) * 0.3
        features[-2] = np.random.uniform(-0.2, 0.5)
    return features


# ════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ════════════════════════════════════════════════════════════

# ── Header ───────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.markdown(
    "**Unsupervised Anomaly Detection** using a deep learning Autoencoder "
    "trained on 284,807 real banking transactions."
)
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ About This App")
    st.markdown("""
    This app uses an **Autoencoder Neural Network** to detect
    fraudulent credit card transactions.

    **How it works:**
    1. The model learned what *normal* transactions look like
    2. It tries to reconstruct any new transaction
    3. High reconstruction error = likely fraud

    **Dataset:** 284,807 transactions (0.17% fraud)

    **Model:** Tuned Autoencoder (TensorFlow/Keras)
    """)

    st.divider()
    st.header("📊 Model Performance")
    for model_name, r in RESULTS.items():
        with st.expander(model_name):
            col1, col2 = st.columns(2)
            col1.metric("AUC-ROC", f"{r['auc_roc']:.4f}")
            col2.metric("AUC-PR",  f"{r['auc_pr']:.4f}")
            col1.metric("Recall",  f"{r['recall']:.2%}")
            col2.metric("Precision", f"{r['precision']:.2%}")

    st.divider()
    st.caption("Built with TensorFlow & Streamlit")
    st.caption("Dataset: Kaggle Credit Card Fraud")


# ── Main Tabs ─────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔮 Predict Transaction",
    "📊 Model Results",
    "📖 How It Works"
])


# ════════════════════════════════════════
# TAB 1 — PREDICTION
# ════════════════════════════════════════
with tab1:
    st.subheader("Analyse a Transaction")

    # Quick demo buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    demo_normal = col1.button("▶ Load Normal Transaction", type="secondary")
    demo_fraud  = col2.button("⚠️ Load Suspicious Transaction", type="secondary")

    # Session state for features
    if 'features' not in st.session_state:
        st.session_state.features = np.zeros(INPUT_DIM)

    if demo_normal:
        st.session_state.features = generate_random_transaction(fraud=False)
        st.success("✅ Normal transaction loaded!")

    if demo_fraud:
        st.session_state.features = generate_random_transaction(fraud=True)
        st.warning("⚠️ Suspicious transaction loaded!")

    st.markdown("#### Transaction Features")
    st.caption(
        "V1–V28 are PCA-transformed features (anonymised by the bank). "
        "Adjust the sliders or use the demo buttons above."
    )

    # Feature sliders — show only most important ones
    key_features = {
        'V1': (-10.0, 10.0),  'V2': (-10.0, 10.0),
        'V3': (-10.0, 10.0),  'V4': (-10.0, 10.0),
        'V10': (-10.0, 10.0), 'V11': (-10.0, 10.0),
        'V14': (-15.0, 10.0), 'V17': (-15.0, 10.0),
        'Amount_scaled': (-1.0, 20.0),
        'Time_scaled': (-3.0, 3.0),
    }

    updated_features = st.session_state.features.copy()
    feature_idx_map  = {name: i for i, name in enumerate(FEATURE_NAMES)}

    cols = st.columns(2)
    for i, (fname, (fmin, fmax)) in enumerate(key_features.items()):
        if fname in feature_idx_map:
            idx = feature_idx_map[fname]
            cur = float(np.clip(updated_features[idx], fmin, fmax))
            updated_features[idx] = cols[i % 2].slider(
                fname, fmin, fmax, cur, 0.01, key=f"slider_{fname}"
            )

    st.divider()

    # Predict button
    if st.button("🔍 Analyse This Transaction", type="primary", use_container_width=True):
        with st.spinner("Running through Autoencoder..."):
            result = predict_transaction(updated_features)

        st.divider()
        col_result, col_score = st.columns([1, 1])

        with col_result:
            if result['is_fraud']:
                st.markdown(
                    '<div class="fraud-box">🚨 FRAUD DETECTED</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="normal-box">✅ NORMAL TRANSACTION</div>',
                    unsafe_allow_html=True
                )
            st.markdown(f"**Confidence:** {result['confidence']}%")
            st.progress(result['confidence'] / 100)

        with col_score:
            st.markdown("#### 📏 Reconstruction Error")
            st.metric(
                label     = "Error Score",
                value     = f"{result['error']:.6f}",
                delta     = f"Threshold: {result['threshold']:.6f}",
                delta_color = "inverse"
            )
            ratio = result['error'] / result['threshold']
            st.markdown(f"**Error / Threshold ratio:** `{ratio:.2f}x`")
            if result['is_fraud']:
                st.error(f"Error is **{ratio:.1f}x** above threshold → Fraud")
            else:
                st.success(f"Error is **{1/ratio:.1f}x** below threshold → Normal")

        # Feature importance
        st.divider()
        st.markdown("#### 🔑 Which Features Triggered The Alert?")
        st.caption("Features with highest reconstruction error drove this decision.")

        top_idx    = result['top_features']
        top_names  = [FEATURE_NAMES[i] for i in top_idx]
        top_errors = [result['feature_errors'][i] for i in top_idx]
        max_err    = max(top_errors) if top_errors else 1

        for name, err in zip(top_names, top_errors):
            col_n, col_b = st.columns([1, 3])
            col_n.markdown(f"**{name}**")
            col_b.progress(float(err / max_err))


# ════════════════════════════════════════
# TAB 2 — MODEL RESULTS
# ════════════════════════════════════════
with tab2:
    st.subheader("📊 Full Model Comparison")

    import pandas as pd

    rows = []
    for mname, r in RESULTS.items():
        f1 = round(2 * r['precision'] * r['recall'] /
                   (r['precision'] + r['recall'] + 1e-9), 4)
        rows.append({
            'Model'    : mname,
            'AUC-ROC'  : r['auc_roc'],
            'AUC-PR'   : r['auc_pr'],
            'Recall'   : r['recall'],
            'Precision': r['precision'],
            'F1 Score' : f1,
            'Fraud Caught': f"{r['caught']}/492"
        })

    result_df = pd.DataFrame(rows).set_index('Model')
    st.dataframe(
        result_df.style
            .highlight_max(subset=['AUC-ROC','AUC-PR','Recall','F1 Score'],
                           color='#d5f5e3')
            .format({'AUC-ROC': '{:.4f}', 'AUC-PR': '{:.4f}',
                     'Recall': '{:.2%}', 'Precision': '{:.2%}',
                     'F1 Score': '{:.4f}'}),
        use_container_width=True
    )

    st.divider()
    st.markdown("#### 💰 Business Impact Estimate")
    st.caption("Assumptions: avg fraud = $122, false alarm review = $5")

    biz_cols = st.columns(len(RESULTS))
    for i, (mname, r) in enumerate(RESULTS.items()):
        prevented   = r['caught'] * 122
        fa_count    = max(0, int(r['caught'] / (r['precision'] + 1e-9))
                         - r['caught'])
        fa_cost     = fa_count * 5
        net         = prevented - fa_cost
        with biz_cols[i]:
            st.markdown(f"**{mname}**")
            st.metric("Fraud Prevented", f"${prevented:,.0f}")
            st.metric("False Alarm Cost", f"-${fa_cost:,.0f}")
            st.metric("Net Benefit", f"${net:,.0f}",
                       delta="best" if mname == 'Autoencoder (Tuned)' else None)


# ════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ════════════════════════════════════════
with tab3:
    st.subheader("📖 How Anomaly Detection Works")

    st.markdown("### The Core Problem")
    st.markdown(
        "Only **0.17%** of transactions are fraud. A model that labels "
        "everything 'normal' would be 99.83% accurate — yet completely useless."
    )

    st.markdown("### Why Unsupervised Learning?")
    st.markdown(
        "In real banking, fraud patterns evolve constantly. New fraud types "
        "have no labels yet. We train the model on **normal transactions only**, "
        "so it learns to flag anything that deviates."
    )

    st.markdown("### The Autoencoder Approach")
    st.code(
        "Normal transaction → Autoencoder → Near-perfect copy → LOW error  ✅\n"
        "Fraud transaction  → Autoencoder → Distorted copy    → HIGH error 🚨",
        language=None
    )

    st.markdown("### Three Models Compared")
    import pandas as pd
    models_table = pd.DataFrame({
        "Model"      : ["Isolation Forest", "One-Class SVM", "Autoencoder ⭐"],
        "Strategy"   : [
            "Anomalies are easy to isolate",
            "Draw a boundary around normal data",
            "Learn to reconstruct normal transactions"
        ],
        "Best For"   : ["Fast baseline", "Medium datasets", "Best performance"]
    })
    st.table(models_table)

    st.markdown("### Key Metrics — Why Not Accuracy?")
    metrics_table = pd.DataFrame({
        "Metric"    : ["AUC-ROC", "AUC-PR", "Recall", "Precision"],
        "Meaning"   : [
            "Overall ranking ability — 1.0 is perfect, 0.5 is random",
            "Better metric for heavily imbalanced data like fraud",
            "% of real fraud cases the model actually caught",
            "% of fraud alerts that are genuinely fraud"
        ],
        "Priority"  : ["High", "High", "Most Important ⭐", "Medium"]
    })
    st.table(metrics_table)

    st.markdown("### Architecture")
    st.code(
        "Input [30] → Encoder [16→8] → Bottleneck [4] → Decoder [8→16] → Output [30]\n"
        "                                    ↑\n"
        "                     Forces model to learn the ESSENCE\n"
        "                     of normal data, not memorise it",
        language=None
    )

    st.info(
        "💡 The bottleneck is the key — with only 4 neurons, the model "
        "cannot memorise every transaction. It must learn the underlying "
        "pattern of normal behaviour. Fraud breaks that pattern.",
        icon="💡"
    )