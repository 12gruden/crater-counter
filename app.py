import base64
from collections import Counter
from PIL import Image, ImageDraw
import requests
import streamlit as st

st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📦 Skener přepravek")

# Выбор способа загрузки
upload_option = st.radio(
    "Vyberte způsob:", ["Vyfotit fotoaparatem", "Nahrát fotku ze zařízení"]
)

img_file = None
if upload_option == "Vyfotit fotoaparatem":
  img_file = st.camera_input("Vyfotte paletu s přepravkami")
else:
  img_file = st.file_uploader("Nahrát obrázek", type=["jpg", "jpeg", "png"])

# Палитра контрастных цветов для 5+ видов ящиков
COLORS = ["#00FF00", "#FF3333", "#00BFFF", "#FFFF00", "#FF00FF", "#FFA500"]

if img_file is not None:
  try:
    img = Image.open(img_file)
    
    # Оптимизируем размер фото для быстрой отправки без потери качества
    img.thumbnail((1500, 1500))
    img.save("temp.jpg")

    with st.spinner("Počítám přepravky..."):
      with open("temp.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

      # Оптимальные параметры детекции: отсекаем мусор (40%), не склеиваем ряды (50%)
      url = "https://detect.roboflow.com/cbl_crates/4?api_key=PP79RD363i1TjHyPScet&confidence=40&overlap=50"
      headers = {"Content-Type": "application/x-www-form-urlencoded"}

      response = requests.post(url, data=encoded_string, headers=headers)
      result = response.json()

      boxes = result.get("predictions", [])

      if len(boxes) == 0:
        st.image(img, caption="Zpracovávaná fotografie", use_container_width=True)
        st.warning(
            "📦 Přepravky nebyly nalezeny. Zkuste vyfotit z bližší vzdálenosti za lepšího světla."
        )
      else:
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)

        # Распределяем цвета для найденных классов
        class_colors = {}
        color_idx = 0
        for box in boxes:
          label = box["class"]
          if label not in class_colors:
            class_colors[label] = COLORS[color_idx % len(COLORS)]
            color_idx += 1

        # Адаптивная толщина рамки в зависимости от размера фото
        line_width = max(2, int(img.width * 0.005))

        # Отрисовка рамок и подписей
        for box in boxes:
          x, y, w, h = box["x"], box["y"], box["width"], box["height"]
          label = box["class"]
          conf = box.get("confidence", 0)
          color = class_colors[label]

          x0, y0 = x - w / 2, y - h / 2
          x1, y1 = x + w / 2, y + h / 2

          draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
          draw.text((x0 + 6, y0 + 6), f"{label} ({int(conf*100)}%)", fill=color)

        st.image(
            draw_img,
            caption="Detekované přepravky (výsledek)",
            use_container_width=True,
        )

        classes = [p.get("class", "unknown") for p in boxes if isinstance(p, dict)]
        counts = Counter(classes)

        st.subheader("Výsledek:")
        
        # Вывод статистики с цветными маркерами
        for crate_type, total in counts.items():
          color_hex = class_colors.get(crate_type, "#00FF00")
          st.markdown(
              f"<h4><span style='color:{color_hex}'>■</span> {crate_type}: {total} ks</h4>",
              unsafe_allow_html=True
          )
          
        st.info(f"**Celkem nalezeno**: {len(boxes)} ks")

  except Exception as e:
    st.error(f"⚠️ Došlo k chybě při zpracování: {repr(e)}")
