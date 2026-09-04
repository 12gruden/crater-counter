import os
import io
import base64
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont

# ==========================================
# 1. NASTAVENÍ ROBOFLOW A NOVÉHO MODELU (v12)
# ==========================================
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "PP79RD363i1TjHyPScet")
WORKSPACE_NAME = "evgeniya-kurbatova"
PROJECT_NAME = "cbl_crates"
MODEL_VERSION = 12

st.set_page_config(page_title="Počítadlo přepravek RF-DETR", page_icon="📦", layout="wide")
st.title("📦 Automatické počítání přepravek (RF-DETR Small)")

# Палитра цветов (RGB) для 5 видов ящиков
COLOR_PALETTE = {
    0: (0, 220, 100), # Зеленый
    1: (30, 144, 255), # Ярко-синий
    2: (255, 215, 0), # Золотисто-желтый
    3: (255, 127, 80), # Оранжево-коралловый
    4: (186, 85, 211), # Фиолетовый
}

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
                # 1. Буферизация и base64 кодирование
                buffer = io.BytesIO()
                processed_image.save(buffer, format="JPEG", quality=95)
                img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                
                # 2. Запрос к Roboflow API
                api_url = f"https://detect.roboflow.com/{PROJECT_NAME}/{MODEL_VERSION}?api_key={ROBOFLOW_API_KEY}&confidence={int(confidence_threshold * 100)}"
                
                response = requests.post(
                    api_url,
                    data=img_base64,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                
                if response.status_code == 200:
                    predictions = response.json().get("predictions", [])
                    
                    draw_img = processed_image.copy()
                    draw = ImageDraw.Draw(draw_img)
                    
                    # Разделение по типам (классам)
                    class_counts = {}
                    class_color_map = {}
                    known_classes = []
                    
                    for p in predictions:
                        class_name = p.get("class", "Crate")
                        if class_name not in known_classes:
                            known_classes.append(class_name)
                        
                        class_idx = known_classes.index(class_name)
                        color = COLOR_PALETTE.get(class_idx % len(COLOR_PALETTE), (255, 0, 0))
                        class_color_map[class_name] = color
                        
                        # Подсчет количества каждого вида
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
                        
                        # Координаты рамки
                        x, y, w, h = p['x'], p['y'], p['width'], p['height']
                        left = x - w / 2
                        top = y - h / 2
                        right = x + w / 2
                        bottom = y + h / 2
                        
                        # Рисуем контур
                        draw.rectangle([left, top, right, bottom], outline=color, width=3)
                        
                        # Рисуем маленькую плашку с именем класса сверху
                        text_box_height = 14
                        draw.rectangle([left, max(0, top - text_box_height), left + len(class_name) * 8 + 6, top], fill=color)
                        draw.text((left + 3, max(0, top - text_box_height) + 1), class_name, fill=(255, 255, 255))
                    
                    st.image(draw_img, use_container_width=True)
                    st.success(f"🎉 Celkem spočítáno přepravek: **{len(predictions)}**")
                    
                    # Вывод подробной детализации по типам ящиков
                    if class_counts:
                        st.markdown("### 📊 Detail podle typů přepravek:")
                        for c_name, count in class_counts.items():
                            rgb = class_color_map[c_name]
                            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
                            st.markdown(
                                f"<span style='color:{hex_color}; font-weight:bold;'>■ {c_name}:</span> **{count} ks**", 
                                unsafe_allow_html=True
                            )
                else:
                    st.error(f"Chyba API Roboflow ({response.status_code}): {response.text}")
                    
            except Exception as e:
                st.error(f"Chyba při zpracování: {e}")
