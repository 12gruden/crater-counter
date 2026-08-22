import streamlit as st
from roboflow import Roboflow
from collections import Counter
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📦 Skener přepravek")

# Загрузка 4-й версии модели из Roboflow (теперь она задеплоена!)
@st.cache_resource
def load_model():
    rf = Roboflow(api_key="PP79RD363i1TjHyPScet")
    project = rf.workspace("evgeniya-kurbatova").project("cbl_crates")
    return project.version(4).model

try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Chyba připojení k modelu: {repr(e)}")
    model = None

# Выбор: камера или загрузка файла из галереи
upload_option = st.radio("Vyberte způsob:", ["Vyfotit fotoaparatem", "Nahrát fotку ze zařízení"])

img_file = None
if upload_option == "Vyfotit fotoaparatem":
    img_file = st.camera_input("Vyfotte paletu s přepravkami")
else:
    img_file = st.file_uploader("Nahrát obrázek", type=["jpg", "jpeg", "png"])

if img_file and model:
    # Сохраняем фото
    img = Image.open(img_file)
    img.save("temp.jpg")

    with st.spinner("Počítám přepravky..."):
        try:
            # Получаем результат от нейросети
            result = model.predict("temp.jpg", confidence=40, overlap=25)
            
            # Безопасное извлечение данных
            data = result.json() if hasattr(result, "json") else result
            boxes = data.get("predictions", []) if isinstance(data, dict) else []
            
            # Вывод результатов на чешском
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
