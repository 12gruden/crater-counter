import os
import io
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from roboflow import Roboflow

# ==========================================
# 1. NASTAVENÍ ROBOFLOW A NOVÉHO MODELU (v12)
# ==========================================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "PP79RD363i1TjHyPScet")
WORKSPACE_NAME = "evgeniya-kurbatova"
PROJECT_NAME = "cbl_crates"
MODEL_VERSION = 12

st.set_page_config(page_title="Počítadlo přepravek RF-DETR", page_icon="📦", layout="wide")
st.title("📦 Automatické počítání přepravek (RF-DETR Small)")

@st.cache_resource
def load_roboflow_project():
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    return rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)

# ==========================================
# 2. OPTIMALIZACE OBRÁZKU ("EFEKT SNÍMKU OBRAZOVKY")
# ==========================================
def optimize_image_for_detection(image: Image.Image, max_dimension: int = 1280) -> Image.Image:
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    width, height = image.size
    if max(width, height) > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    image = image.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.15)
    
    return image

# ==========================================
# 3. ROZHRANÍ A LOGIKA
# ==========================================
input_method = st.radio(
    "Vyberte způsob vložení obrázku:",
    ("Nahrát soubor", "Použít fotoaparát"),
    horizontal=True
)

uploaded_file = None

if input_method == "Nahrát soubor":
    uploaded_file = st.file_uploader("Nahrajte fotografii palety s přepravkami", type=["jpg", "jpeg", "png", "webp"])
else:
    uploaded_file = st.camera_input("Pořiďte snímek palety")

confidence_threshold = st.sidebar.slider("Páh spolehlivosti (Confidence)", 10, 90, 30, 5) / 100.0

if uploaded_file is not None:
    raw_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Optimalizovaný snímek")
        with st.spinner("Optimalizace ostrosti a velikosti..."):
            processed_image = optimize_image_for_detection(raw_image)
            st.image(processed_image, use_container_width=True)
            
    with col2:
        st.subheader("Výsledek detekce")
        with st.spinner("Analýza novým modelem RF-DETR..."):
            try:
                project = load_roboflow_project()
                version = project.version(MODEL_VERSION)
                
                temp_path = "temp_optimized.jpg"
                processed_image.save(temp_path, quality=95)
                
                # Поддержка обертки predict для любых типов моделей SDK
                if hasattr(version, 'model') and version.model is not None:
                    prediction = version.model.predict(temp_path, confidence=int(confidence_threshold * 100))
                    prediction.save("prediction.jpg")
                    st.image("prediction.jpg", use_container_width=True)
                    total_crates = len(prediction.json().get("predictions", []))
                else:
                    # Запасной вариант вызова напрямую через клиент проекта
                    res = project.predict(temp_path, confidence=int(confidence_threshold * 100)).json()
                    preds = res.get("predictions", [])
                    
                    draw_img = processed_image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    for p in preds:
                        x, y, w, h = p['x'], p['y'], p['width'], p['height']
                        draw.rectangle([x - w/2, y - h/2, x + w/2, y + h/2], outline="red", width=3)
                    
                    st.image(draw_img, use_container_width=True)
                    total_crates = len(preds)
                
                st.success(f"🎉 Spočítáno přepravek: **{total_crates}**")
                
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
            except Exception as e:
                st.error(f"Chyba při zpracování: {e}")
