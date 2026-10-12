import os
from autodistill.detection import CaptionOntology
from autodistill_grounding_dino import GroundingDINO

# 1. 設定文字提示 (Caption -> 類別名稱)
ontology = CaptionOntology(
    {
        "person": "person",  # 找人
        "surfboard": "surfboard",  # 找衝浪板/風浪板
        "boat": "boat",  # 找船
        "kayak": "kayak",  # 找獨木舟
    }
)

# 2. 載入模型
base_model = GroundingDINO(ontology=ontology)

# 3. 指定照片資料夾
image_dir = os.path.expanduser("~/testpic")

# 4. 開始自動標注
base_model.label(input_folder=image_dir, extension=".jpg")

print("✅ Grounding DINO 自動標注完成！")
