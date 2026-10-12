"""
RT-DETRv4: Painlessly Furthering Real-Time Object Detection with Vision Foundation Models
Copyright (c) 2025 The RT-DETRv4 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
"""  # 專案版權宣告與歷史修正說明字串

from .rtv4 import RTv4  # 匯入 RT-DETRv4 核心整體模型架構主體

from .matcher import HungarianMatcher  # 匯入匈牙利匹配器 (Hungarian Matcher)，用於預測框與真值框的二分圖匹配
from .hybrid_encoder import HybridEncoder  # 匯入原版 3 尺度 (P3~P5) 混合編碼器 (Neck 模組)
from .dfine_decoder import DFINETransformer  # 匯入 D-FINE Transformer 解碼器模組，用於特徵解碼與邊界框回歸
from .rtdetrv2_decoder import RTDETRTransformerv2  # 匯入 RT-DETRv2 Transformer 解碼器備選模組

from .postprocessor import PostProcessor  # 匯入後處理器，用於將解碼器輸出轉為最終檢測邊界框與類別得分
from .rtv4_criterion import RTv4Criterion  # 匯入 RT-DETRv4 損失函數計算模組 (Criterion)

from .dinov3_teacher import DINOv3TeacherModel  # 匯入 DINOv3 教師模型，用於知識蒸餾 (Distillation) 訓練
from .hybrid_encoder_P2 import HybridEncoderP2  # 匯入自定義支援 4 尺度 (P2~P5) 的 P2 專用混合編碼器模組
from . import hybrid_encoder_P2  # 補上這行，確保模組本身也能被系統反射 (Reflection) 找到
