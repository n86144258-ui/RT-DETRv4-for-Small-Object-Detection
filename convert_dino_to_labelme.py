import json
import os
from PIL import Image

# 照片所在資料夾
IMAGES_DIR = os.path.expanduser("~/testpic")

# 標注來源資料夾（同時包含 train 與 valid）
LABEL_DIRS = [
    os.path.expanduser("~/testpic_labeled/train/labels"),
    os.path.expanduser("~/testpic_labeled/valid/labels"),
]

# 物件類別順序 (對應 dino_auto_label.py)
CLASSES = ["person", "surfboard", "boat", "kayak"]

count = 0

for label_dir in LABEL_DIRS:
    if not os.path.exists(label_dir):
        continue

    for txt_file in os.listdir(label_dir):
        if not txt_file.endswith(".txt"):
            continue

        txt_path = os.path.join(label_dir, txt_file)
        base_name = os.path.splitext(txt_file)[0]

        # 尋找對應的照片副檔名 (.jpg 或 .png)
        img_name = base_name + ".jpg"
        img_path = os.path.join(IMAGES_DIR, img_name)

        if not os.path.exists(img_path):
            img_name = base_name + ".png"
            img_path = os.path.join(IMAGES_DIR, img_name)
            if not os.path.exists(img_path):
                continue

        # 讀取照片長寬
        with Image.open(img_path) as img:
            width, height = img.size

        shapes = []
        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            x_center = float(parts[1]) * width
            y_center = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height

            # 計算左上角與右下角座標
            x1 = x_center - (w / 2)
            y1 = y_center - (h / 2)
            x2 = x_center + (w / 2)
            y2 = y_center + (h / 2)

            label_name = (
                CLASSES[cls_id] if cls_id < len(CLASSES) else f"class_{cls_id}"
            )

            shapes.append(
                {
                    "label": label_name,
                    "points": [[x1, y1], [x2, y2]],
                    "group_id": None,
                    "description": "",
                    "shape_type": "rectangle",
                    "flags": {},
                }
            )

        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }

        json_path = os.path.join(IMAGES_DIR, base_name + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)

        count += 1

print(
    f"✅ 成功將全部 {count} 張照片（含 train + valid）的標注轉換完成，全數存入 ~/testpic！"
)
