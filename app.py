

import streamlit as st
from roboflow import Roboflow
from collections import Counter
from PIL import Image

# Nastavení stránky
st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📦 Skener přepravek")

# Načtení modelu z Roboflow
@st.cache_resource
def load_model():
    
    rf = Roboflow(api_key="PP79RD363i1TjHyPScet") 
    project = rf.workspace("evgeniya-kurbatova").project("cbl_crates")
    return project.version(3).model

try:
    model = load_model()
except Exception:
    st.error("Chyba připojení k modelu. Zkontrolujte API Key.")

# Tlačítko pro fotoaparát
img_file = st.camera_input("Vyfoťte paletu s přepravkami")

if img_file:
    img = Image.open(img_file)
    img.save("temp.jpg")
    
    with st.spinner("Počítám přepravky..."):
        # Detekce ящиков
        prediction = model.predict("temp.jpg", confidence=40, overlap=25).json()
        
        # Подсчет результатов
        classes = [p["class"] for p in prediction["predictions"]]
        counts = Counter(classes)
        
        st.subheader("Výsledek:")
        if counts:
            for crate_type, total in counts.items():
                st.success(f"**{crate_type}**: {total} ks")
            st.info(f"**Celkem nalezeno**: {len(classes)} ks")
        else:
            st.warning("Přepravky nebyly nalezeny. Zkuste vyfotit z bližší vzdálenosti.")
