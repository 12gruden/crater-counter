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
      # Кодируем картинку в base64 для отправки через requests
      with open("temp.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

      payload = {
          "api_key": "PP79RD363i1TjHyPScet",
          "inputs": {"image": {"type": "base64", "value": encoded_string}},
      }

      # Отправляем запрос на Workflow API Roboflow напрямую
      response = requests.post(
          "https://detect.roboflow.com/infer/workflows/evgeniya-kurbatova/cbl_crates-vcblcrates-4-yolo11n-t1-logic",
          json=payload,
      )
      result = response.json()

      # Безопасно извлекаем предсказания из ответа
      boxes = []
      if isinstance(result, dict):
        for k, v in result.items():
          if isinstance(v, list):
            boxes = v
            break
          elif isinstance(v, dict):
            if "predictions" in v:
              boxes = v["predictions"]
              break
            for sub_k, sub_v in v.items():
              if isinstance(sub_v, list):
                boxes = sub_v
                break

      valid_boxes = []
      for item in boxes:
        if isinstance(item, dict):
          if "class" in item:
            valid_boxes.append(item)
          elif "predictions" in item:
            valid_boxes.extend(item["predictions"])

      if len(valid_boxes) == 0:
        st.warning(
            "📦 Přepravky nebyly nalezeny. Zkuste vyfotit z bližší vzdálenosti."
        )
      else:
        classes = [
            p.get("class", "unknown") for p in valid_boxes if isinstance(p, dict)
        ]
        counts = Counter(classes)

        st.subheader("Výsledek:")
        for crate_type, total in counts.items():
          st.success(f"**{crate_type}**: {total} ks")
        st.info(f"**Celkem nalezeno**: {len(valid_boxes)} ks")

  except Exception as e:
    st.error(f"⚠️ Došlo k chybě při zpracování: {repr(e)}")
