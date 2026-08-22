import streamlit as st
from inference_sdk import InferenceHTTPClient
from collections import Counter
from PIL import Image

st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📦 Skener přepravek")

# Подключение к Workflow через Inference SDK
@st.cache_resource
def get_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="PP79RD363i1TjHyPScet"
    )

client = None
try:
    with st.spinner("Načítám model..."):
        client = get_client()
    st.success("Model je připraven!")
except Exception as e:
    st.error(f"⚠️ Chyba připojení k API: {repr(e)}")
    client = None

# Выбор способа (камера или загрузка файла)
upload_option = st.radio("Vyberte způsob:", ["Vyfotit fotoaparatem", "Nahrát fotku ze zařízení"])

img_file = None
if upload_option == "Vyfotit fotoaparatem":
    img_file = st.camera_input("Vyfotte paletu s přepravkami")
else:
    img_file = st.file_uploader("Nahrát obrázek", type=["jpg", "jpeg", "png"])

if img_file is not None:
    if client is None:
        st.error("⚠️ Model není načten, nelze zpracovat fotku.")
    else:
        try:
            # Открываем и сжимаем фото
            img = Image.open(img_file)
            img.thumbnail((1024, 1024))
            img.save("temp.jpg")
            
            # Показываем превью картинки
            st.image(img, caption="Zpracovávaná fotografie", use_container_width=True)

            with st.spinner("Počítám přepravky..."):
                # Запуск воркфлоу
                result = client.run_workflow(
                    workspace_name="evgeniya-kurbatova",
                    workflow_id="cbl_crates-vcblcrates-4-yolo11n-t1-logic",
                    images={
                        "image": "temp.jpg"
                    }
                )
                
                # Безопасный поиск предсказаний в ответе воркфлоу
                boxes = []
                if isinstance(result, list) and len(result) > 0:
                    for item in result:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) and "class" in v[0]:
                                    boxes = v
                                    break
                                elif isinstance(v, dict) and "predictions" in v:
                                    boxes = v["predictions"]
                                    break
                elif isinstance(result, dict):
                    boxes = result.get("predictions", [])

                if len(boxes) == 0:
                    st.warning("📦 Přepravky nebyly nalezeny. Zkuste vyfotit z bližší vzdálenosti.")
                else:
                    classes = [p.get("class", "unknown") for p in boxes if isinstance(p, dict)]
                    counts = Counter(classes)
                    
                    st.subheader("Výsledek:")
                    for crate_type, total in counts.items():
                        st.success(f"**{crate_type}**: {total} ks")
                    st.info(f"**Celkem nalezeno**: {len(boxes)} ks")
                    
        except Exception as e:
            st.error(f"⚠️ Došlo k chybě při zpracování: {repr(e)}")
