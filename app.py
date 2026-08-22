import streamlit as st
from inference_sdk import InferenceHTTPClient
from collections import Counter
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📦 Skener přepravek")

# Инициализация клиента Roboflow Workflow
@st.cache_resource
def get_workflow_client():
    return InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key="PP79RD363i1TjHyPScet"
    )

try:
    client = get_workflow_client()
except Exception as e:
    st.error(f"⚠️ Chyba připojení k API: {repr(e)}")
    client = None

# Кнопка для камеры телефона/ноута
img_file = st.camera_input("Vyfotte paletu s přepravkami")

if img_file and client:
    # Сохраняем фото
    img = Image.open(img_file)
    img.save("temp.jpg")

    with st.spinner("Počítám přepravky..."):
        try:
            # Запуск воркфлоу через Inference SDK
            result = client.run_workflow(
                workspace_name="evgeniya-kurbatova",
                workflow_id="cbl_crates-vcblcrates-4-yolo11n-t1-logic",
                images={
                    "image": "temp.jpg"
                }
            )
            
            # Универсальный поиск предсказаний в ответе воркфлоу
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

            # Вывод результатов на чешском языке
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
