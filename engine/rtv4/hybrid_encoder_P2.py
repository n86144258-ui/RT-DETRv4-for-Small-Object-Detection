import torch  # 匯入 PyTorch 核心套件
import torch.nn as nn  # 匯入 PyTorch 的神經網路模組 (nn)
import torch.nn.functional as F  # 匯入 PyTorch 的功能性函數庫 (如插值、激活函數)

from engine.core import register  # 修正為 RT-DETRv4 專案內部的註冊器路徑

__all__ = ['HybridEncoderP2']  # 定義對外暴露的類別名稱清單

class ConvNormLayer(nn.Module):  # 定義包含 卷積 + 歸一化 + 激活函數 的基礎組合模組
    def __init__(self, ch_in, ch_out, filter_size, stride, padding=None, act=None):  # 初始化函數與輸入參數設定
        super().__init__()  # 呼叫父類別 nn.Module 的初始化函數
        if padding is None:  # 若未指定填充值 (padding)
            padding = (filter_size - 1) // 2  # 自動計算 padding 確保特徵圖尺寸符合預期
        self.conv = nn.Conv2d(ch_in, ch_out, filter_size, stride=stride, padding=padding, bias=False)  # 建立 2D 卷積層 (不使用偏置項)
        self.norm = nn.BatchNorm2d(ch_out)  # 建立 2D 批次歸一化層 (Batch Normalization)
        self.act = nn.SiLU() if act == 'silu' else (nn.ReLU() if act == 'relu' else nn.Identity())  # 根據參數選擇 SiLU、ReLU 或不加激活函數

    def forward(self, x):  # 前向傳播函數
        return self.act(self.norm(self.conv(x)))  # 依序執行：卷積 -> 歸一化 -> 激活函數 並返回結果


class RepVGGBlock(nn.Module):  # 定義 RepVGG 風格的重參數化特徵提取區塊
    def __init__(self, ch_in, ch_out, act='silu'):  # 初始化函數
        super().__init__()  # 呼叫父類別 nn.Module 的初始化函數
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, act=act)  # 建立 3x3 卷積分支
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, act=act)  # 建立 1x1 卷積分支

    def forward(self, x):  # 前向傳播函數
        return self.conv1(x) + self.conv2(x)  # 將 3x3 分支與 1x1 分支的特徵相加融合


@register()  # 使用註冊器裝飾器，讓 YAML 設定檔能透過名稱找到這個類別
class HybridEncoderP2(nn.Module):  # 定義支援 P2 特徵層的 4 尺度混合編碼器類別
    def __init__(  # 初始化函數與預設參數設定
        self,  # 指向物件實體本身
        in_channels=[128, 256, 512, 1024],  # 輸入特徵圖的通道數清單，依次對應 [P2, P3, P4, P5]
        feat_strides=[4, 8, 16, 32],  # 特徵圖相較原圖的縮放步長/倍率清單 [1/4, 1/8, 1/16, 1/32]
        hidden_dim=256,  # 編碼器內部的統一特徵隱藏維度 (通道數)
        nhead=8,  # Transformer 多頭注意力機制的頭數 (Multi-Head Attention)
        dim_feedforward=1024,  # Transformer 前饋網路 (FFN) 的前向擴展維度
        dropout=0.0,  # Dropout 機率 (預設 0.0 表示不隨機丟棄神經元)
        enc_act='gelu',  # Transformer 編碼器層採用的激活函數類型
        use_encoder_idx=[3],  # 指定自注意力機制 (AIFI) 作用的特徵層索引 (3 代表最高層 P5)
        num_encoder_layers=1,  # Transformer 編碼器堆疊的層數
        pe_temperature=10000,  # 位置編碼 (Positional Encoding) 的溫度係數
        expansion=1.0,  # 通道擴展係數 (用於控制模型體積)
        depth_mult=1.0,  # 網路深度乘數 (用於控制模型層數)
        act='silu',  # 卷積區塊所使用的激活函數類型
        eval_spatial_size=None,  # 評估階段固定的空間解析度尺寸 (若無則設為 None)
    ):  # 初始化參數宣告結束
        super().__init__()  # 呼叫父類別 nn.Module 的初始化函數
        self.in_channels = in_channels  # 將輸入通道數清單存為成員變數
        self.feat_strides = feat_strides  # 將步長清單存為成員變數
        self.hidden_dim = hidden_dim  # 將隱藏維度存為成員變數
        self.use_encoder_idx = use_encoder_idx  # 將 Encoder 作用索引存為成員變數
        self.num_levels = len(in_channels)  # 計算輸入特徵層的數量 (此處應為 4 個層級)

        # 1. 建立通道投影層列表：將 [P2, P3, P4, P5] 不同的輸入通道數統一映射到 hidden_dim (如 256)
        self.input_proj = nn.ModuleList([  # 建立 ModuleList 容器儲存多個投影層
            nn.Sequential(  # 針對每個輸入層級建立順序網路
                nn.Conv2d(in_channel, hidden_dim, 1, bias=False),  # 使用 1x1 卷積改變通道數
                nn.BatchNorm2d(hidden_dim)  # 對投影後的特徵圖進行批次歸一化
            ) for in_channel in in_channels  # 針對 [P2, P3, P4, P5] 的每個通道數進行迭代迴圈
        ])  # 通道投影層列表建立完成

        # 2. 建立 Transformer 自注意力編碼器單元 (AIFI)
        encoder_layer = nn.TransformerEncoderLayer(  # 建立單層 Transformer 編碼器結構
            d_model=hidden_dim,  # 輸入與輸出的維度 (256)
            nhead=nhead,  # 注意力機制頭數 (8)
            dim_feedforward=dim_feedforward,  # FFN 中間隱藏層維度 (1024)
            dropout=dropout,  # Dropout 機率 (0.0)
            activation=enc_act,  # 激活函數 ('gelu')
            batch_first=True  # 設定 Tensor 張量的 Batch 維度在第 0 維度 (batch, seq, feature)
        )  # 單層 Transformer 編碼器定義完成
        self.aifi = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)  # 堆疊指定層數的 Transformer 編碼器模組

        # 3. 建立自頂向下 (Top-Down) 融合區塊列表 (共需要 3 次融合：P5->P4, P4->P3, P3->P2)
        self.top_down_blocks = nn.ModuleList([  # 建立 ModuleList 容器
            RepVGGBlock(hidden_dim, hidden_dim, act=act) for _ in range(3)  # 建立 3 個 RepVGG 特徵融合區塊
        ])  # 自頂向下融合區塊建立完成

        # 4. 建立自底向上 (Bottom-Up) 下採樣卷積與融合區塊列表 (共需要 3 次融合：P2->P3, P3->P4, P4->P5)
        self.downsample_convs = nn.ModuleList([  # 建立下採樣卷積層列表
            ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act) for _ in range(3)  # 建立 3 個 步長為 2 的 3x3 下採樣卷積
        ])  # 下採樣卷積層建立完成
        self.bottom_up_blocks = nn.ModuleList([  # 建立自底向上融合區塊列表
            RepVGGBlock(hidden_dim, hidden_dim, act=act) for _ in range(3)  # 建立 3 個 RepVGG 特徵融合區塊
        ])  # 自底向上融合區塊建立完成

    def forward(self, feats):  # 前向傳播函數，傳入 Backbone 提取的特徵圖清單 [P2, P3, P4, P5]
        assert len(feats) == self.num_levels, f"Expected {self.num_levels} inputs, got {len(feats)}"  # 斷言檢查傳入的特徵層數量是否為 4

        # 步驟 A: 將 4 個層級的特徵圖進行 1x1 卷積投影，統一通道數至 hidden_dim
        proj_feats = [self.input_proj[i](feats[i]) for i in range(self.num_levels)]  # 獲得通道調整後的 [proj_P2, proj_P3, proj_P4, proj_P5]

        # 步驟 B: 將最高層特徵圖 (P5，索引 3) 送入 AIFI 進行全局自注意力 (Self-Attention) 計算
        p5 = proj_feats[3]  # 取得最高層特徵圖 proj_P5
        bs, c, h, w = p5.shape  # 取得 proj_P5 的維度大小 (Batch, Channel, Height, Width)
        p5_flat = p5.flatten(2).permute(0, 2, 1)  # 將空間維度 (H, W) 展平為長度為 H*W 的序列，並調整形狀為 (bs, H*W, c)
        p5_aifi = self.aifi(p5_flat).permute(0, 2, 1).reshape(bs, c, h, w)  # 送入 AIFI 計算後，再恢復成 2D 特徵圖形狀 (bs, c, h, w)
        proj_feats[3] = p5_aifi  # 用經過 AIFI 增強後的 P5 特徵圖替換原來的 proj_P5

        # 步驟 C: Top-Down 自頂向下特徵融合路徑 (P5 -> P4 -> P3 -> P2)
        inner_outs = [proj_feats[3]]  # 初始化內部特徵列表，第一個元素放增強後的 P5
        for idx in range(3, 0, -1):  # 迴圈控制倒序進行：idx 依次為 3 (P5), 2 (P4), 1 (P3)
            feat_high = inner_outs[0]  # 取得當前較高層級的特徵圖
            feat_low = proj_feats[idx - 1]  # 取得下一個較低層級的特徵圖
            upsample_feat = F.interpolate(feat_high, scale_factor=2, mode='nearest')  # 將高層特徵圖放大 2 倍 (上採樣) 對齊解析度
            fused = self.top_down_blocks[3 - idx](upsample_feat + feat_low)  # 將上採樣特徵與低層特徵相加後，通過 RepVGG 區塊融合
            inner_outs.insert(0, fused)  # 將融合後的特徵插入到列表最前端，順序最後會變成 [fused_P2, fused_P3, fused_P4, fused_P5]

        # 步驟 D: Bottom-Up 自底向上特徵融合路徑 (P2 -> P3 -> P4 -> P5)
        outs = [inner_outs[0]]  # 融合後的 fused_P2 直接作為第 0 層的最終輸出特徵圖
        for idx in range(3):  # 迴圈控制順序進行：idx 依次為 0, 1, 2
            feat_low = outs[-1]  # 取得前一次計算好的較低層級輸出特徵圖
            feat_high = inner_outs[idx + 1]  # 取得下一層級 Top-Down 階段的中間特徵圖
            downsampled_feat = self.downsample_convs[idx](feat_low)  # 將低層特徵圖經由步長 2 卷積縮小 2 倍 (下採樣)
            fused = self.bottom_up_blocks[idx](downsampled_feat + feat_high)  # 將下採樣特徵與高層特徵相加後，通過 RepVGG 區塊融合
            outs.append(fused)  # 將融合結果追加到輸出清單中

        # 最終返回 4 個尺度的輸出特徵圖列表 [out_P2, out_P3, out_P4, out_P5]
        return outs  # 前向傳播結束，返回融合後的特徵圖結果
