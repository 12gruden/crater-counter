import base64
from collections import Counter
from PIL import Image, ImageDraw
import requests
import streamlit as st
import io

st.set_page_config(page_title="Skener přepravek", layout="centered")
st.title("📸 Skener přepravek")

# Výběr způsobu nahrání
upload_option = st.radio(
    "Vyberte způsob:", ["Vyfotit fotoaparátem", "Nahrát fotku ze zařízení"]
)

img_file = None
if upload_option == "Vyfotit fotoaparátem":
    img_file = st.camera_input("Pořiďte snímek")
else:
    img_file = st.file_uploader("Nahrát obrázek", type=["jpg", "jpeg", "png"])

# Paleta kontrastních barev
COLORS = ["#00FF00", "#FF3333", "#00BFFF", "#FFFF00", "#FF00FF", "#FFA500"]

if img_file is not None:
    try:
        img = Image.open(img_file)
        
        # Optimalizace velikosti pro odeslání
        img.thumbnail((1500, 1500))

        # Zpracování obrázku přímo v paměti (bez ukládání temp.jpg na disk)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        base64_encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")

        with st.spinner("Počítám přepravky..."):
            
            # VERZE 5 - Optimalizované parametry
            url = "https://detect.roboflow.com/cbl_crates/5?api_key=PP79RD363i1TjHyPScet&confidence=40&overlap=50"
            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            response = requests.post(url, data=base64_encoded, headers=headers)
            result = response.json()

            boxes = result.get("predictions", [])

            if len(boxes) == 0:
                st.image(img, caption="Původní fotografie", use_container_width=True)
                st.warning(
                    "📸 Přepravky nebyly nalezeny. Pokuste se pořídit snímek z menší vzdálenosti nebo za lepšího osvětlení."
                )
            else:
                draw_img = img.copy()
                draw = ImageDraw.Draw(draw_img)

                # Rozdělení barev pro nalezené třídy
                class_colors = {}
                color_idx = 0
                for box in boxes:
                    label = box["class"]
                    if label not in class_colors:
                        class_colors[label] = COLORS[color_idx % len(COLORS)]
                        color_idx += 1

                # Adaptivní tloušťka rámečku
                line_width = max(2, int(img.width * 0.005))

                # Kreslení rámečků
                for box in boxes:
                    x, y, w, h = box["x"], box["y"], box["width"], box["height"]
                    label = box["class"]
                    conf = box.get("confidence", 0)
                    color = class_colors[label]

                    x0, y0 = x - w / 2, y - h / 2
                    x1, y1 = x + w / 2, y + h / 2

                    draw.rectangle([x0, y0, x1, y1], outline=color, width=line_width)
                    draw.text((x0 + 6, y0 + 6), f"{label} {int(conf*100)}%", fill=color)

                st.image(
                    draw_img,
                    caption="Detekované přepravky (výsledek)",
                    use_container_width=True,
                )

                classes = [p.get("class", "neznámý") for p in boxes if isinstance(p, dict)]
                counts = Counter(classes)

                st.subheader("Seznam:")
                
                # Výpis statistiky
                for crate_type, count in counts.items():
                    color_hex = class_colors.get(crate_type, "#00FF00")
                    st.markdown(
                        f"<h4><span style='color:{color_hex}'>■</span> {crate_type}: {count} ks</h4>",
                        unsafe_allow_html=True,
                    )

                st.info(f"**Celkem nalezeno**: {len(boxes)} ks")

    except Exception as e:
        st.error(f"Došlo k chybě při zpracování: {repr(e)}")
