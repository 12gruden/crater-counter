

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

  if picture:
        # Сохраняем фото
        with open("temp.jpg", "wb") as file:
            file.write(picture.getvalue())
        
        try:
            # Получаем результат от нейросети
            result = model.predict("temp.jpg", confidence=40, overlap=25)
            
            # Безопасно извлекаем данные (защита от пустой стены)
            data = result.json() if hasattr(result, "json") else result
            
            # Достаем список найденных ящиков
            boxes = data.get("predictions", [])
            
            if len(boxes) == 0:
                # Желтое предупреждение на чешском: "На этой фото не найдены ящики. Попробуйте сфотографировать иначе."
                st.warning("📦 Na této fotce nebyly nalezeny žádné přepravky. Zkuste to vyfotit jinak.")
            else:
                # Зеленое сообщение об успехе на чешском: "Найдено ящиков: X"
                st.success(f"✅ Nalezeno přepravek: {len(boxes)}")
                
                # Рисуем рамки
                image = Image.open("temp.jpg")
                draw = ImageDraw.Draw(image)
                
                for box in boxes:
                    x0 = box['x'] - box['width'] / 2
                    y0 = box['y'] - box['height'] / 2
                    x1 = box['x'] + box['width'] / 2
                    y1 = box['y'] + box['height'] / 2
                    
                    draw.rectangle([x0, y0, x1, y1], outline="red", width=4)
                
                # Показываем результат
                st.image(image, use_container_width=True)
                
        except Exception as e:
            # Красное сообщение об ошибке (если пропадет интернет или сбой): "Произошла ошибка обработки. Попробуйте снова."
            st.error("⚠️ Došlo k chybě při zpracování. Zkuste to prosím znovu.")


