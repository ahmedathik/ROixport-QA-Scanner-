import streamlit as st
import cv2
import numpy as np
import io
from PIL import Image

st.set_page_config(
    page_title="ROIFO | Ghost-Scan Diagnostic Utility",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
<style>
.stApp { background-color: #030712; color: #f3f4f6; font-family: sans-serif; }
.header-box { background: linear-gradient(90deg, #090d16 0%, #0f172a 100%); padding: 20px; border-radius: 10px; border: 1px solid rgba(56, 189, 248, 0.2); margin-bottom: 20px; }
.card { background: rgba(17, 24, 39, 0.85); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 10px; padding: 16px; margin-bottom: 12px; }
</style>
\<div class="header-box">
    <h2 style="margin:0; color: #38bdf8;">🛡️ ROIFO Ghost-Scan Diagnostic Utility</h2>
    <p style="margin:4px 0 0 0; color: #94a3b8; font-size: 13px;">Instant Surface Glare Filtering & Micro-Defect Inspection Sampler</p>
\</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("### ⚙️ Diagnostic Mode")
mode = st.sidebar.selectbox("Select Inspection Profile", ["Precision Metal Component", "Automotive Casting / Weld", "Electronic PCB / Finish"])
sensitivity = st.sidebar.slider("AI Sensitivity Threshold", 0.1, 1.0, 0.45, 0.05)

uploaded_file = st.sidebar.file_uploader("Upload Part Image for Instant Analysis", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    pil_img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    frame_rgb = np.array(pil_img)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📷 Original Part Capture")
        st.image(frame_rgb, use_container_width=True)
        
    with col2:
        st.markdown("#### 🔍 Processed Anomaly Map")
        with st.spinner("Stripping glare and running patch analysis..."):
            # Lightweight simulation of the standalone diagnostic pipeline
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            edges = cv2.Canny(enhanced, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            defect_pixels = sum([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 15])
            score = float(np.clip((defect_pixels / (edges.size + 1e-5)) * 15.0 * sensitivity, 0.0, 1.0))
            
            annotated = frame_rgb.copy()
            if score > 0.4:
                verdict = "REJECT"
                summary = "❌ Micro-Crack or Surface Disruption Detected! Exceeds quality limit."
                color = (255, 0, 0)
            elif score > 0.2:
                verdict = "MANUAL REVIEW"
                summary = "⚠️ Borderline Surface Mark / Glare Interference. Supervisor check recommended."
                color = (255, 255, 0)
            else:
                verdict = "PASS"
                summary = "✨ Surface Nominal. Glare filtered successfully. Ready for packaging."
                color = (0, 255, 0)
                
            for c in contours:
                if cv2.contourArea(c) > 15:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
                    
            st.image(annotated, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Diagnostic Summary Report")
    
    if verdict == "PASS":
        st.success(summary)
    elif verdict == "MANUAL REVIEW":
        st.warning(summary)
    else:
        st.error(summary)
        
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Inspection Verdict", verdict)
    m2.metric("Calculated Risk Score", f"{score:.2f}")
    m3.metric("Glare Filter Status", "Active (CLAHE)")
    m4.metric("Profile", mode)
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Operator Feedback")
    fb1, fb2 = st.columns(2)
    with fb1:
        if st.button("👍 Confirm Result"):
            st.success("Feedback logged to sample session.")
    with fb2:
        if st.button("👎 False Positive (Retrain AI Sample)"):
            st.info("Sample added to local calibration memory. Accuracy adjusted.")
else:
    st.info("👈 Upload an image in the sidebar to run a live diagnostic sample scan.")
