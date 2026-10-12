import json
import os

# 設定檔案路徑
COCO_JSON = "./testpic_labels.json"
IMG_DIR = os.path.expanduser("~/testpic")

with open(COCO_JSON, "r", encoding="utf-8") as f:
    coco = json.load(f)

# 建立類別對照表
cat_map = {cat["id"]: cat["name"] for cat in coco["categories"]}

# 建立每張照片的註解對照
img_anns = {img["id"]: [] for img in coco["images"]}
for ann in coco.get("annotations", []):
    img_anns[ann["image_id"]].append(ann)

# 逐張產生 Labelme 格式的 .json 檔
for img in coco["images"]:
    img_id = img["id"]
    file_name = img["file_name"]
    json_name = os.path.splitext(file_name)[0] + ".json"
    json_path = os.path.join(IMG_DIR, json_name)

    shapes = []
    for ann in img_anns[img_id]:
        # bbox 格式: [x, y, width, height]
        x, y, w, h = ann["bbox"]
        label = cat_map.get(ann["category_id"], "object")

        shapes.append(
            {
                "label": label,
                "points": [[x, y], [x + w, y + h]],  # 對角線兩點 (Rectangle)
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
        "imagePath": file_name,
        "imageData": None,
        "imageHeight": img["height"],
        "imageWidth": img["width"],
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(labelme_data, f, indent=2, ensure_ascii=False)

print(
    f"✅ 成功將標注檔轉換並寫入 {IMG_DIR}！現在可以用 Labelme 直接打開修改了。"
)
