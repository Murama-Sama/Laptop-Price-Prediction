import streamlit as st
import pandas as pd
import joblib

# ==========================
# CONFIG (HARUS PALING ATAS)
# ==========================
st.set_page_config(
    page_title="Laptop Price Prediction",
    page_icon="💻",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("laptop_price_model.pkl")
encoders = joblib.load("label_encoders.pkl")

# ==========================
# CSS
# ==========================
st.markdown("""
<style>
.stApp{
    background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
}
h1{
    text-align:center;
    color:white;
}
.stButton>button{
    width:100%;
    height:55px;
    font-size:22px;
    border-radius:15px;
    background:#00C853;
    color:white;
    font-weight:bold;
}
div[data-testid="stMetric"]{
    background:#202020;
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# ==========================
# TITLE
# ==========================
st.title("💻 Laptop Price Prediction")
st.write("Masukkan spesifikasi laptop lalu klik Predict.")

# ==========================
# INPUT
# ==========================
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", encoders["brand"].classes_)
    processor = st.selectbox("Processor", encoders["processor"].classes_)
    cpu = st.selectbox("CPU", encoders["CPU"].classes_)
    ram = st.number_input("RAM (GB)", 2, 64, 8)
    ram_type = st.selectbox("RAM Type", encoders["Ram_type"].classes_)
    rom = st.number_input("ROM (GB)", 128, 4000, 512)
    rom_type = st.selectbox("ROM Type", encoders["ROM_type"].classes_)

with col2:
    gpu = st.selectbox("GPU", encoders["GPU"].classes_)
    display = st.number_input("Display Size", 10.0, 20.0, 15.6)
    width = st.number_input("Resolution Width", 800, 4000, 1920)
    height = st.number_input("Resolution Height", 600, 3000, 1080)
    os = st.selectbox("OS", encoders["OS"].classes_)
    warranty = st.number_input("Warranty (Years)", 0, 5, 1)
    spec = st.slider("Specification Rating", 0.0, 100.0, 75.0)

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.title("💻 Laptop Predictor")
    st.write("---")
    st.info("""
Model: Gradient Boosting Regressor  
Dataset: Laptop Price Dataset  
R² Score: 0.88
""")

# ==========================
# PREDICTION
# ==========================
st.write("---")

if st.button("🔍 Predict Price"):

    try:
        data = pd.DataFrame([{
            "brand": encoders["brand"].transform([brand])[0],
            "spec_rating": spec,
            "processor": encoders["processor"].transform([processor])[0],
            "CPU": encoders["CPU"].transform([cpu])[0],
            "Ram": ram,
            "Ram_type": encoders["Ram_type"].transform([ram_type])[0],
            "ROM": rom,
            "ROM_type": encoders["ROM_type"].transform([rom_type])[0],
            "GPU": encoders["GPU"].transform([gpu])[0],
            "display_size": display,
            "resolution_width": width,
            "resolution_height": height,
            "OS": encoders["OS"].transform([os])[0],
            "warranty": warranty
        }])

        hasil = model.predict(data)[0]

        st.success("Prediksi berhasil!")
        st.metric("💰 Predicted Price", f"Rp {hasil:,.0f}")

    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")

# ==========================
# FOOTER
# ==========================
st.caption("Developed by Raihan • Machine Learning Project • 2026")