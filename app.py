import base64
from collections import Counter
from PIL import Image
import requests
import streamlit as st

st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📦 Skener přepravek")

# Выбор способа (камера или загрузка файла)
upload_option = st.radio(
    "Vyberte způsob:", ["Vyfotit fotoaparatem", "Nahrát fotku ze zařízení"]
)

img_file = None
if upload_option == "Vyfotit fotoaparatem":
  img_file = st.camera_input("Vyfotte paletu s přepravkami")
else:
  img_file = st.file_uploader("Nahrát obrázek", type=["jpg", "jpeg", "png"])

if img_file is not None:
  try:
    # Открываем и сжимаем фото
    img = Image.open(img_file)
    img.thumbnail((1024, 1024))
    img.save("temp.jpg")

    # Показываем превью картинки
    st.image(img, caption="Zpracovávaná fotografie", use_container_width=True)

    with st.spinner("Počítám přepravky..."):
      # Кодируем картинку в base64
      with open("temp.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

      url = "https://detect.roboflow.com/cbl_crates/4?api_key=PP79RD363i1TjHyPScet"
      
      # Добавляем нужный Content-Type заголовок
      headers = {"Content-Type": "application/x-www-form-urlencoded"}

      # Отправляем запрос с заголовками
      response = requests.post(url, data=encoded_string, headers=headers)
      result = response.json()

      # Извлекаем предсказания модели
      boxes = result.get("predictions", [])

      if len(boxes) == 0:
        st.warning(
            "📦 Přepravky nebyly nalezeny. Zkuste vyfotit z bližší vzdálenosti"
            " nebo zkontrolujte osvětlení."
        )
        st.write("Odpověd modelu:", result)
      else:
        classes = [
            p.get("class", "unknown") for p in boxes if isinstance(p, dict)
        ]
        counts = Counter(classes)

        st.subheader("Výsledek:")
        for crate_type, total in counts.items():
          st.success(f"**{crate_type}**: {total} ks")
        st.info(f"**Celkem nalezeno**: {len(boxes)} ks")

  except Exception as e:
    st.error(f"⚠️ Došlo k chybě při zpracování: {repr(e)}")
