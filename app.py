import streamlit as st
from roboflow import Roboflow
from collections import Counter
from PIL import Image

st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📦 Skener přepravek")

@st.cache_resource
def load_model():
    rf = Roboflow(api_key="PP79RD363i1TjHyPScet")
    project = rf.workspace("evgeniya-kurbatova").project("cbl_crates")
    return project.version(4).model

# Загружаем модель с отловом ошибок
model = None
try:
    with st.spinner("Načítám model..."):
        model = load_model()
    st.success("Model je připraven!")
except Exception as e:
    st.error(f"⚠️ Chyba připojení k modelu: {repr(e)}")

# Выбор способа
upload_option = st.radio("Vyberte způsob:", ["Vyfotit fotoaparatem", "Nahrát fotku ze zařízení"])

img_file = None
if upload_option == "Vyfotit fotoaparatem":
    img_file = st.camera_input("Vyfotte paletu s přepravkami")
else:
    img_file = st.file_uploader("Nahrát obrázek", type=["jpg", "jpeg", "png"])

if img_file is not None:
    if model is None:
        st.error("⚠️ Model není načten, nelze zpracovat fotku.")
    else:
        try:
            # Открываем фото и сжимаем его до безопасного размера (чтобы 6.5 МБ не вешали приложение)
            img = Image.open(img_file)
            img.thumbnail((1024, 1024))
            img.save("temp.jpg")
            
            # Показываем миниатюру загруженного фото
            st.image(img, caption="Zpracovávaná fotografie", use_container_width=True)

            with st.spinner("Počítám přepravky..."):
                result = model.predict("temp.jpg", confidence=40, overlap=25)
                
                data = result.json() if hasattr(result, "json") else result
                boxes = data.get("predictions", []) if isinstance(data, dict) else []
                
                if len(boxes) == 0:
                    st.warning("📦 Přepravky nebyly nalezeny. Zkuste vyfotit z bližší vzdálenosti.")
                else:
                    classes = [p["class"] for p in boxes if "class" in p]
                    counts = Counter(classes)
                    
                    st.subheader("Výsledek:")
                    for crate_type, total in counts.items():
                        st.success(f"**{crate_type}**: {total} ks")
                    st.info(f"**Celkem nalezeno**: {len(boxes)} ks")
                    
        except Exception as e:
            st.error(f"⚠️ Došlo k chybě při zpracování: {repr(e)}")
