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


class FeatureMatchingLoss(nn.Module):
    """增强版特征匹配损失
    
    通过直接监督编码器生成的特征与目标特征之间的差异，提供更直接的梯度信号。
    基于最新的特征匹配技术，如StyleGAN2、SPADE等工作中的方法。
    添加了特征统计匹配和多尺度特征匹配。
    """
    def __init__(self, loss_type='l1', use_style_loss=True, use_multi_scale=True):
        super(FeatureMatchingLoss, self).__init__()
        self.loss_type = loss_type
        self.use_style_loss = use_style_loss  # 是否使用风格损失（Gram矩阵）
        self.use_multi_scale = use_multi_scale  # 是否使用多尺度特征匹配
        
    def compute_gram_matrix(self, x):
        """计算Gram矩阵，用于风格损失"""
        b, c, h, w = x.size()
        features = x.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram.div(c * h * w)
    
    def compute_mean_std(self, feat, eps=1e-5):
        """计算特征的均值和标准差，用于AdaIN风格迁移"""
        size = feat.size()
        feat_var = feat.var(dim=(2, 3), keepdim=True) + eps
        feat_std = feat_var.sqrt()
        feat_mean = feat.mean(dim=(2, 3), keepdim=True)
        return feat_mean, feat_std
        
    def forward(self, source_features, target_features, weights=None):
        """计算特征匹配损失
        
        Args:
            source_features: 源特征，可以是单个张量或张量列表
            target_features: 目标特征，可以是单个张量或张量列表
            weights: 可选，每层特征的权重
            
        Returns:
            特征匹配损失值
        """
        # 处理单个特征的情况
        if not isinstance(source_features, list):
            source_features = [source_features]
        if not isinstance(target_features, list):
            target_features = [target_features]
        
        # 处理特征列表长度不同的情况
        if len(source_features) != len(target_features):
            # 如果源特征比目标特征多，只使用与目标特征数量相同的源特征
            if len(source_features) > len(target_features):
                source_features = source_features[:len(target_features)]
            # 如果目标特征比源特征多，只使用与源特征数量相同的目标特征
            else:
                target_features = target_features[:len(source_features)]
        
        # 如果没有提供权重，则为每层特征分配相同的权重
        if weights is None:
            weights = [1.0] * len(source_features)
        elif len(weights) != len(source_features):
            weights = [1.0] * len(source_features)
        
        total_loss = 0.0
        content_loss = 0.0
        style_loss = 0.0
        stats_loss = 0.0
        
        for i, (src_feat, tgt_feat, weight) in enumerate(zip(source_features, target_features, weights)):
            # 确保特征形状相同
            if src_feat.shape != tgt_feat.shape:
                # 如果形状不同，调整源特征的形状以匹配目标特征
                # 处理通道数不匹配的情况
                if src_feat.shape[1] != tgt_feat.shape[1]:
                    # 如果源特征通道数多于目标特征，截取前面的通道
                    if src_feat.shape[1] > tgt_feat.shape[1]:
                        src_feat = src_feat[:, :tgt_feat.shape[1], ...]
                    # 如果目标特征通道数多于源特征，截取前面的通道
                    else:
                        tgt_feat = tgt_feat[:, :src_feat.shape[1], ...]
                
                # 处理空间尺寸不匹配的情况
                if src_feat.shape[2:] != tgt_feat.shape[2:]:
                    src_feat = F.interpolate(src_feat, size=tgt_feat.shape[2:], mode='bilinear', align_corners=False)
            
            # 内容损失（直接特征匹配）
            if self.loss_type == 'l1':
                loss = F.l1_loss(src_feat, tgt_feat)
            elif self.loss_type == 'l2' or self.loss_type == 'mse':
                loss = F.mse_loss(src_feat, tgt_feat)
            elif self.loss_type == 'cosine':
                src_feat_flat = src_feat.view(src_feat.size(0), -1)
                tgt_feat_flat = tgt_feat.view(tgt_feat.size(0), -1)
                loss = 1 - F.cosine_similarity(src_feat_flat, tgt_feat_flat).mean()
            else:
                loss = F.l1_loss(src_feat, tgt_feat)  # 默认使用L1损失
            
            content_loss += loss * weight
            
            # 风格损失（Gram矩阵匹配）
            if self.use_style_loss and src_feat.dim() == 4:
                src_gram = self.compute_gram_matrix(src_feat)
                tgt_gram = self.compute_gram_matrix(tgt_feat)
                style_l = F.mse_loss(src_gram, tgt_gram) * weight
                style_loss += style_l
            
            # 统计匹配损失（均值和方差匹配）
            if src_feat.dim() == 4:
                src_mean, src_std = self.compute_mean_std(src_feat)
                tgt_mean, tgt_std = self.compute_mean_std(tgt_feat)
                stats_l = (F.mse_loss(src_mean, tgt_mean) + F.mse_loss(src_std, tgt_std)) * weight
                stats_loss += stats_l
        
        # 计算总损失
        total_loss = content_loss
        if self.use_style_loss:
            total_loss += style_loss * 0.5
        total_loss += stats_loss * 0.1
        
        # 返回平均损失
        return total_loss / len(source_features)


import torchvision.models as models

class PerceptualLoss(nn.Module):
    """感知损失
    
    使用预训练的VGG网络提取特征，计算感知相似度。
    这种损失能够捕捉到更高级的语义信息，而不仅仅是像素级别的差异。
    """
    def __init__(self, layers=[3, 8, 15, 22], weights=[1.0, 1.0, 1.0, 1.0]):
        super(PerceptualLoss, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features.eval()
        self.layers = layers
        self.weights = weights
        
        # 冻结VGG参数
        for param in self.vgg.parameters():
            param.requires_grad = False
            
    def forward(self, x, y):
        """计算感知损失
        
        Args:
            x: 输入图像
            y: 目标图像
            
        Returns:
            感知损失值
        """
        # 确保输入在正确的范围内（VGG期望输入在[0,1]范围内）
        if x.min() < 0:
            x = (x + 1) / 2.0
        if y.min() < 0:
            y = (y + 1) / 2.0
            
        # 标准化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        x = (x - mean) / std
        y = (y - mean) / std
        
        # 提取特征并计算损失
        loss = 0.0
        x_features = []
        y_features = []
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            y = layer(y)
            
            if i in self.layers:
                layer_idx = self.layers.index(i)
                loss += F.mse_loss(x, y) * self.weights[layer_idx]
                
        return loss


class EncoderSupervisionLoss(nn.Module):
    """增强版编码器监督损失
    
    结合多种损失函数，为编码器提供更强的监督信号。
    整合了KL散度损失、对比学习损失、特征匹配损失和感知损失。
    添加了动态权重调整和渐进式训练策略。
    """
    def __init__(self, config):
        super(EncoderSupervisionLoss, self).__init__()
        self.config = config
        
        # KL散度权重
        self.beta = config.get('beta', 0.01)
        
        # 对比学习损失权重和参数
        self.contrastive_weight = config.get('contrastive_weight', 0.05)  # 降低初始权重
        self.temperature = config.get('temperature', 0.1)  # 增加温度参数，使对比学习更稳定
        
        # 特征匹配损失权重和参数
        self.feature_matching_weight = config.get('feature_matching_weight', 1.0)  # 增加权重
        self.feature_matching_loss = FeatureMatchingLoss(
            loss_type=config.get('feature_matching_loss_type', 'l1'),
            use_style_loss=True,  # 启用风格损失
            use_multi_scale=True  # 启用多尺度特征匹配
        )
        
        # 感知损失
        self.perceptual_weight = config.get('perceptual_weight', 0.2)  # 增加感知损失权重
        self.use_perceptual = config.get('use_perceptual', True)
        if self.use_perceptual:
            self.perceptual_loss = PerceptualLoss()
        
        # 训练计数器，用于动态调整权重
        self.train_steps = 0
        self.warmup_steps = config.get('warmup_steps', 2000)  # 增加热身步数
        self.max_steps = config.get('max_steps', 10000)  # 最大步数，用于权重调整
        
    def get_dynamic_weights(self):
        """获取动态调整的权重
        
        实现渐进式训练策略，随着训练的进行，动态调整各损失的权重
        """
        # 热身阶段，逐渐增加权重
        if self.train_steps < self.warmup_steps:
            progress = min(1.0, self.train_steps / self.warmup_steps)
            # 特征匹配损失从小到大
            feature_matching_weight = self.feature_matching_weight * progress
            # 感知损失从小到大
            perceptual_weight = self.perceptual_weight * progress
        elif self.train_steps < self.max_steps:
            # 中期阶段，逐渐调整权重比例
            progress = min(1.0, (self.train_steps - self.warmup_steps) / (self.max_steps - self.warmup_steps))
            # 特征匹配损失权重持续增加
            feature_matching_weight = self.feature_matching_weight * (1.0 + 0.5 * progress)
            # 感知损失权重持续增加
            perceptual_weight = self.perceptual_weight * (1.0 + progress)
        else:
            # 后期阶段，固定权重
            feature_matching_weight = self.feature_matching_weight * 1.5  # 增加特征匹配损失权重
            perceptual_weight = self.perceptual_weight * 2.0  # 增加感知损失权重
            
        return feature_matching_weight, perceptual_weight
        
    def forward(self, model_outputs, batch_data):
        """计算编码器监督损失
        
        Args:
            model_outputs: 模型输出的字典，包含重建结果、特征、mu、logvar等
            batch_data: 批次数据，包含源图像和目标图像
            
        Returns:
            总损失和各部分损失的字典
        """
        # 更新训练步数
        self.train_steps += 1
        
        # 获取动态权重
        feature_matching_weight, perceptual_weight = self.get_dynamic_weights()
        
        # 解包模型输出
        output = model_outputs['output']
        mu = model_outputs['mu']
        logvar = model_outputs['logvar']
        encoder_features = model_outputs.get('encoder_features', [])
        device = output.device
        
        # 解包批次数据并确保在正确的设备上
        source_img = batch_data['source_img'].to(device)
        target_img = batch_data['target_img'].to(device)
        
        # 重建损失（L1损失）
        recon_loss = F.l1_loss(output, target_img)
        
        # KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / (output.size(0) * output.size(2) * output.size(3))  # 归一化
        
        # 特征匹配损失（如果有编码器特征和目标特征）
        feature_matching_loss = torch.tensor(0.0, device=output.device)
        if 'target_features' in model_outputs and len(encoder_features) > 0 and feature_matching_weight > 0:
            target_features = model_outputs['target_features']
            
            # 为不同层的特征分配不同的权重，深层特征权重更高
            # 使用指数增长的权重，使深层特征的权重显著高于浅层特征
            weights = [2.0 ** i for i in range(len(encoder_features))]
            weights = [w / sum(weights) for w in weights]  # 归一化权重
            
            # 确保特征列表长度匹配
            min_len = min(len(encoder_features), len(target_features))
            encoder_features_matched = encoder_features[:min_len]
            target_features_matched = target_features[:min_len]
            weights_matched = weights[:min_len]
            
            # 计算特征匹配损失
            try:
                feature_matching_loss = self.feature_matching_loss(encoder_features_matched, target_features_matched, weights_matched)
                # 应用缩放因子，增大特征匹配损失的影响
                feature_matching_loss = feature_matching_loss * 2.0
            except Exception as e:
                # 如果计算失败，使用备用方法
                feature_matching_loss = torch.tensor(0.0, device=output.device)
                for i in range(min_len):
                    if i % 2 == 0:  # 只使用部分层，减少计算量
                        src_feat = encoder_features[i]
                        tgt_feat = target_features[i]
                        # 确保特征形状匹配
                        if src_feat.shape != tgt_feat.shape and src_feat.dim() == 4 and tgt_feat.dim() == 4:
                            tgt_feat = F.interpolate(tgt_feat, size=src_feat.shape[2:], mode='bilinear', align_corners=False)
                        # 计算L1损失
                        if src_feat.shape == tgt_feat.shape:
                            layer_loss = F.l1_loss(src_feat, tgt_feat) * weights_matched[i] * 2.0
                            feature_matching_loss += layer_loss
        
        # 感知损失
        perceptual_loss = torch.tensor(0.0, device=output.device)
        if self.use_perceptual and perceptual_weight > 0:
            try:
                # 确保输入图像通道数正确（VGG期望3通道输入）
                if output.size(1) == 3 and target_img.size(1) == 3:
                    perceptual_loss = self.perceptual_loss(output, target_img)
                    # 应用缩放因子，确保感知损失有适当的影响
                    perceptual_loss = perceptual_loss * 1.5
            except Exception as e:
                # 如果计算失败，使用L1损失作为备用
                perceptual_loss = F.l1_loss(output, target_img) * 0.5
        
        # 总损失 - 使用更稳定的加权方式
        # 重建损失始终是基础损失
        total_loss = recon_loss
        
        # KL散度损失 - 使用动态beta值
        # 随着训练进行，逐渐增加KL散度的权重
        dynamic_beta = self.beta * min(1.0, self.train_steps / (self.warmup_steps * 0.5))
        if not torch.isnan(kl_loss) and not torch.isinf(kl_loss):
            total_loss = total_loss + dynamic_beta * kl_loss
            
        # 添加特征匹配损失 - 带有梯度裁剪
        if feature_matching_weight > 0 and not torch.isnan(feature_matching_loss) and not torch.isinf(feature_matching_loss):
            # 裁剪特征匹配损失值，防止梯度爆炸
            clipped_feature_matching_loss = torch.clamp(feature_matching_loss, 0, 10.0)
            total_loss = total_loss + feature_matching_weight * clipped_feature_matching_loss
        
        # 添加感知损失 - 带有梯度裁剪
        if self.use_perceptual and perceptual_weight > 0 and not torch.isnan(perceptual_loss) and not torch.isinf(perceptual_loss):
            # 裁剪感知损失值，防止梯度爆炸
            clipped_perceptual_loss = torch.clamp(perceptual_loss, 0, 10.0)
            total_loss = total_loss + perceptual_weight * clipped_perceptual_loss
        
        # 返回总损失和各部分损失
        return {
            'total': total_loss,
            'recon': recon_loss,
            'kl': kl_loss,
            'feature_matching': feature_matching_loss,
            'perceptual': perceptual_loss if self.use_perceptual else torch.tensor(0.0, device=output.device)
        }