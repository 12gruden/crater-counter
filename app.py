import base64
from collections import Counter
from PIL import Image, ImageDraw
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
    img = Image.open(img_file)
    img.save("temp.jpg")

    with st.spinner("Počítám přepravky..."):
      with open("temp.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

      # Универсальные рабочие параметры
      url = "https://detect.roboflow.com/cbl_crates/4?api_key=PP79RD363i1TjHyPScet&confidence=40&overlap=30"
      headers = {"Content-Type": "application/x-www-form-urlencoded"}

      response = requests.post(url, data=encoded_string, headers=headers)
      result = response.json()

      boxes = result.get("predictions", [])

      if len(boxes) == 0:
        st.image(
            img, caption="Zpracovávaná fotografie", use_container_width=True
        )
        st.warning(
            "📦 Přepravky nebyly nalezeny. Zkuste vyfotit z bližší vzdálenosti"
            " za lepšího světla."
        )
      else:
        # Создаем копию картинки для отрисовки рамок
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        for box in boxes:
          x = box["x"]
          y = box["y"]
          w = box["width"]
          h = box["height"]
          label = box["class"]
          conf = box.get("confidence", 0)

          # Переводим координаты центра в углы рамки
          x0 = x - w / 2
          y0 = y - h / 2
          x1 = x + w / 2
          y1 = y + h / 2

          # Рисуем рамку и подпись
          draw.rectangle([x0, y0, x1, y1], outline="#00FF00", width=4)
          draw.text((x0 + 6, y0 + 6), f"{label} ({int(conf*100)}%)", fill="#00FF00")

        # Показываем фото с нарисованными рамками
        st.image(
            draw_img,
            caption="Detekované přepravky (výsledek)",
            use_container_width=True,
        )

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
