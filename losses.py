import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """重建损失
    
    用于计算生成图像与目标图像之间的重建损失
    支持多种损失类型：L1、L2/MSE、SSIM和混合损失
    """
    def __init__(self, loss_type='l1', ssim_weight=0.2):
        super(ReconstructionLoss, self).__init__()
        self.loss_type = loss_type
        self.ssim_weight = ssim_weight  # SSIM损失权重，仅在混合损失中使用
        
    def forward(self, x, y):
        """计算重建损失
        
        Args:
            x: 生成图像
            y: 目标图像
            
        Returns:
            重建损失值
        """
        if self.loss_type == 'l1':
            return F.l1_loss(x, y)
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            return F.mse_loss(x, y)
        elif self.loss_type == 'ssim':
            return 1 - self.compute_ssim(x, y)
        elif self.loss_type == 'mixed':
            # 混合L1和SSIM损失
            l1_loss = F.l1_loss(x, y)
            ssim_loss = 1 - self.compute_ssim(x, y)
            return (1 - self.ssim_weight) * l1_loss + self.ssim_weight * ssim_loss
        else:
            # 默认使用L1损失
            return F.l1_loss(x, y)
    
    def compute_ssim(self, x, y, window_size=11, size_average=True):
        """计算SSIM损失
        
        结构相似性指数衡量两个图像之间的感知相似度
        """
        # 检查输入尺寸
        if x.size() != y.size():
            raise ValueError(f"输入尺寸不匹配: {x.size()} vs {y.size()}")
        
        # 窗口函数
        def create_window(window_size, channel):
            _1D_window = torch.ones(window_size) / window_size
            _2D_window = _1D_window.unsqueeze(1).mm(_1D_window.unsqueeze(0))
            window = _2D_window.unsqueeze(0).unsqueeze(0).expand(channel, 1, window_size, window_size)
            return window.to(x.device)
        
        # 获取通道数
        c = x.size(1)
        window = create_window(window_size, c)
        
        # 均值和方差
        mu1 = F.conv2d(x, window, padding=window_size//2, groups=c)
        mu2 = F.conv2d(y, window, padding=window_size//2, groups=c)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=c) - mu1_sq
        sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=c) - mu2_sq
        sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=c) - mu1_mu2
        
        # SSIM公式中的常数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class ContrastiveLoss(nn.Module):
    """对比学习损失
    
    用于多编码器架构中，促使不同编码器的特征相互区分
    基于InfoNCE损失
    """
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, features):
        """计算对比学习损失
        
        Args:
            features: 编码器特征列表
            
        Returns:
            对比学习损失
        """
        if len(features) <= 1:
            return torch.tensor(0.0, device=features[0].device)
        
        batch_size = features[0].shape[0]
        device = features[0].device
        
        # 将特征展平为向量
        flat_features = []
        for feature in features:
            # 确保特征形状一致
            if feature.dim() > 2:
                # 如果是卷积特征图，先进行全局平均池化
                feature = F.adaptive_avg_pool2d(feature, 1).view(batch_size, -1)
            flat_features.append(feature)
        
        # 计算特征之间的相似度矩阵
        similarity_matrix = torch.zeros((batch_size * len(flat_features), batch_size * len(flat_features)), device=device)
        
        for i, feat_i in enumerate(flat_features):
            for j, feat_j in enumerate(flat_features):
                # 计算第i个编码器和第j个编码器的特征之间的相似度
                i_start = i * batch_size
                i_end = (i + 1) * batch_size
                j_start = j * batch_size
                j_end = (j + 1) * batch_size
                
                # 归一化特征
                feat_i_norm = F.normalize(feat_i, p=2, dim=1)
                feat_j_norm = F.normalize(feat_j, p=2, dim=1)
                
                # 计算余弦相似度
                sim = torch.mm(feat_i_norm, feat_j_norm.t()) / self.temperature
                similarity_matrix[i_start:i_end, j_start:j_end] = sim
        
        # 创建标签：对角块为正样本，其他为负样本
        labels = torch.zeros(batch_size * len(flat_features), dtype=torch.long, device=device)
        for i in range(len(flat_features)):
            labels[i * batch_size:(i + 1) * batch_size] = i
        
        # 计算对比损失
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetCentricFeatureMatchingLoss(nn.Module):
    """
    只与目标特征 (默认 key="depth") 做配对的特征匹配损失。

    例 - 输入 features 字典
        {
            "depth":  Tensor(B, C, H, W),   # 目标
            "color":  Tensor(...),
            "canny":  Tensor(...),
            "sketch": Tensor(...)
        }
    forward 会依次计算
        depth ↔ color 、 depth ↔ canny 、 depth ↔ sketch
    的内容 / Gram / 均值方差损失，然后加权求平均 (或总和) 返回。
    """

    def __init__(self,
                 target_key: str = "depth",
                 loss_type: str = "l1",
                 use_style_loss: bool = True,
                 use_stats_loss: bool = True,
                 reduce: str = "mean"       # "sum" or "mean"
                 ):
        super().__init__()
        self.target_key     = target_key
        self.loss_type      = loss_type.lower()
        self.use_style_loss = use_style_loss
        self.use_stats_loss = use_stats_loss
        assert reduce in ("sum", "mean")
        self.reduce = reduce

    # -------- 通用工具 --------
    @staticmethod
    def _gram(x):
        b, c, h, w = x.size()
        feat = x.view(b, c, -1)
        return torch.bmm(feat, feat.transpose(1, 2)) / (c * h * w)

    @staticmethod
    def _mean_std(x, eps=1e-5):
        var  = x.var(dim=(2, 3), keepdim=True, unbiased=False) + eps
        std  = var.sqrt()
        mean = x.mean(dim=(2, 3), keepdim=True)
        return mean, std

    # -------- forward --------
    def forward(self, features: dict[str, torch.Tensor]):
        if self.target_key not in features:
            raise KeyError(f"features 字典中未找到 target_key='{self.target_key}'")

        f_tgt = features[self.target_key]                     # 目标深度特征
        losses = []

        for k, f_src in features.items():
            if k == self.target_key:     # 跳过自己
                continue

            # ---- 形状对齐 ----
            if f_src.shape[1] != f_tgt.shape[1]:
                c = min(f_src.shape[1], f_tgt.shape[1])
                f_src, f_tgt = f_src[:, :c], f_tgt[:, :c]
            if f_src.shape[2:] != f_tgt.shape[2:]:
                f_src = F.interpolate(f_src, size=f_tgt.shape[2:],
                                      mode="bilinear", align_corners=False)

            # ---- 内容损失 ----
            if self.loss_type == "l1":
                pair_loss = F.l1_loss(f_src, f_tgt)
            elif self.loss_type in ("l2", "mse"):
                pair_loss = F.mse_loss(f_src, f_tgt)
            elif self.loss_type == "cosine":
                pair_loss = 1 - F.cosine_similarity(
                    f_src.flatten(1), f_tgt.flatten(1)).mean()
            else:
                pair_loss = F.l1_loss(f_src, f_tgt)

            # ---- Gram 风格 ----
            if self.use_style_loss:
                pair_loss += 0.5 * F.mse_loss(self._gram(f_src), self._gram(f_tgt))

            # ---- 均值 / 方差 ----
            if self.use_stats_loss:
                m1, s1 = self._mean_std(f_src)
                m2, s2 = self._mean_std(f_tgt)
                pair_loss += 0.1 * (F.mse_loss(m1, m2) + F.mse_loss(s1, s2))

            losses.append(pair_loss)

        if self.reduce == "sum":
            return sum(losses)
        else:                            # "mean"
            return sum(losses) / len(losses)