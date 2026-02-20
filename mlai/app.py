import os
# Set Keras backend to torch
os.environ["KERAS_BACKEND"] = "torch"
# Disable TorchDynamo to avoid needing C++ compiler (cl.exe)
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import streamlit as st
import keras
import numpy as np
from PIL import Image
import torch

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Nature Analyzer AI | Natural Scene Classifier",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CLEAN MINIMAL DARK THEME CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Deep Space Background */
    .stApp {
        background: linear-gradient(135deg, #020617 0%, #0f172a 100%);
        color: #f8fafc;
    }

    /* Floating Animation */
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }

    /* Cyber Hero Section */
    .hero-wrapper {
        position: relative;
        padding: 5rem 2rem;
        background: radial-gradient(circle at top center, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
        border-radius: 40px;
        margin-bottom: 4rem;
        overflow: hidden;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }

    .hero-title {
        font-size: 5rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(to right, #818cf8, #c084fc, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.05em;
        margin-bottom: 1rem;
        animation: float 6s ease-in-out infinite;
    }

    .hero-subtitle {
        font-size: 1.4rem;
        font-weight: 300;
        text-align: center;
        color: #94a3b8;
        max-width: 800px;
        margin: 0 auto;
        opacity: 0.8;
    }

    /* Glass Panels */
    .glass-panel {
        background: rgba(15, 23, 42, 0.4);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 32px;
        padding: 2.5rem;
        height: 100%;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .glass-panel:hover {
        border: 1px solid rgba(129, 140, 248, 0.3);
        transform: scale(1.02);
        box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.5);
    }

    /* Neon Prediction Card */
    .neon-result {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(168, 85, 247, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 32px;
        padding: 4rem 2rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }

    .neon-result::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: conic-gradient(from 180deg at 50% 50%, transparent 0%, #6366f1 20%, transparent 100%);
        animation: rotate 10s linear infinite;
        opacity: 0.1;
    }

    @keyframes rotate {
        100% { transform: rotate(360deg); }
    }

    .result-category {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.4em;
        color: #818cf8;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }

    .result-tag {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(to bottom, #fff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        margin: 1rem 0;
        filter: drop-shadow(0 0 20px rgba(99, 102, 241, 0.3));
    }

    /* Animated Confidence Bar */
    .conf-wrap {
        margin-top: 3rem;
        background: rgba(2, 6, 23, 0.5);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }

    .conf-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 1rem;
        color: #cbd5e1;
        font-size: 0.9rem;
        font-weight: 600;
    }

    .conf-rail {
        height: 10px;
        background: rgba(30, 41, 59, 1);
        border-radius: 5px;
        overflow: hidden;
    }

    .conf-train {
        height: 100%;
        background: linear-gradient(90deg, #6366f1, #a855f7, #22d3ee);
        background-size: 200% 100%;
        animation: wave 3s linear infinite;
        border-radius: 5px;
        transition: width 1.5s cubic-bezier(0.19, 1, 0.22, 1);
    }

    @keyframes wave {
        0% { background-position: 100% 0%; }
        100% { background-position: -100% 0%; }
    }

    /* Footer Badges */
    .land-badge {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        color: #64748b;
        padding: 8px 20px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
        margin: 5px;
        transition: all 0.3s ease;
    }

    .land-badge:hover {
        background: rgba(99, 102, 241, 0.1);
        color: #818cf8;
        border-color: rgba(99, 102, 241, 0.3);
    }

    /* Streamlit Overrides */
    .stFileUploader section {
        background: rgba(15, 23, 42, 0.3) !important;
        border: 2px dashed rgba(129, 140, 248, 0.2) !important;
        border-radius: 24px !important;
    }

    [data-testid="stHeader"] {
        background: rgba(2, 6, 23, 0.8) !important;
        backdrop-filter: blur(12px) !important;
    }
</style>


""", unsafe_allow_html=True)

# --- MODEL CORE ---
@st.cache_resource
def load_trained_model():
    model_path = r"best_cnn_model.keras"
    if os.path.exists(model_path):
        try:
            with torch.no_grad():
                return keras.models.load_model(model_path)
        except Exception as e:
            st.error(f"Intelligence Core Initialization Error: {e}")
            return None
    return None

model = load_trained_model()
class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

# --- INTERFACE ---
st.markdown("""
    <div class="hero-wrapper">
        <h1 class="hero-title">NATURE ANALYZER AI PRO</h1>
        <p class="hero-subtitle">Next-Generation Neural Scene Intelligence Engine<br>
        <span style="font-weight: 600; color: #818cf8;">Architect: Jasna Jaison</span></p>
    </div>
""", unsafe_allow_html=True)

main_col1, main_col2 = st.columns([1, 1], gap="medium")

with main_col1:
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown("### 💠 NEURAL INPUT")
    uploaded_file = st.file_uploader("Upload satellite or natural landscape imagery", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, use_container_width=True)
    else:
        st.write("---")
        st.markdown('<p style="color:#64748b; text-align:center; font-style: italic;">Awaiting telemetry data...</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with main_col2:
    if uploaded_file:
        if model:
            with st.spinner("Analyzing spectral data..."):
                # Inference
                img = img.convert('RGB')
                img_p = img.resize((150, 150))
                img_arr = np.array(img_p).astype('float32') / 255.0
                img_arr = np.expand_dims(img_arr, axis=0)
                
                with torch.no_grad():
                    preds = model.predict(img_arr, verbose=0)
                
                # Probs
                def softmax(x):
                    e = np.exp(x - np.max(x))
                    return e / e.sum()
                
                probs = softmax(preds[0]) if (np.sum(preds[0]) > 1.01 or np.sum(preds[0]) < 0.99) else preds[0]
                idx = np.argmax(probs)
                conf = probs[idx] * 100
                
                # --- PREMIUM RESULT ---
                st.markdown(f"""
                    <div class="neon-result">
                        <div class="result-category">Neural Classification Result</div>
                        <div class="result-tag">{class_names[idx].upper()}</div>
                        <div class="conf-wrap">
                            <div class="conf-label">
                                <span>Confidence Accuracy</span>
                                <span>{conf:.2f}%</span>
                            </div>
                            <div class="conf-rail">
                                <div class="conf-train" style="width: {conf}%"></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                with st.expander("Detailed Probability Distribution"):
                    for i, name in enumerate(class_names):
                        p = probs[i] * 100
                        col_a, col_b = st.columns([3, 1])
                        col_a.write(f"**{name.capitalize()}**")
                        col_b.write(f"{p:.1f}%")
                        st.progress(int(p))
        else:
            st.error("Model Error: Neural weights file missing.")
    else:
        st.markdown("""
            <div style="height: 100%; min-height: 300px; display: flex; align-items: center; justify-content: center; border: 1px dashed #30363d; border-radius: 12px; background-color: #0d1117;">
                <p style="color: #484f58;">Classification Report Generator</p>
            </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #475569; font-weight: 600; letter-spacing: 0.1em;">RECOGNIZED LANDSCAPES</p>', unsafe_allow_html=True)
cols = st.columns(6)
for i, cls in enumerate(class_names):
    cols[i].markdown(f'<div class="land-badge" style="width: 100%; text-align: center;">{cls.upper()}</div>', unsafe_allow_html=True)

st.markdown(f"""
    <div style="margin-top: 5rem; color: #475569; font-size: 0.8rem; text-align: center; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 2rem;">
        Neural Processor v3.0.4 • <b>Jasna Jaison</b> • Edge Compute Engine • 2026
    </div>
""", unsafe_allow_html=True)
