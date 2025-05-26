# 标题
# CRUCIBLE: Conditional Representation Unification through Cross-modal Integration with Bottlenecked Latent Encoding
### CRUCIBLE本意"熔炉"，隐喻工作本质：将各种视觉条件熔炼在一起，提取其共同本质，形成统一的表征。
### 就像冶金师将不同金属熔于一炉，提纯去杂，最终锻造出具有卓越性能的合金。缩写CRUCIBLE既朗朗上口，又直接唤起融合与提纯的形象。
### a place or situation in which different cultures or styles can mix together to produce something new and exciting

# 1. 摘要

# 2. 引言
基于视觉引导的图像生成能够通过各种条件类型（例如草图、深度图和语义分割）实现精确控制。然而，随着视觉条件多样性的不断扩展，
生成模型面临着一个关键挑战：每种新的条件类型都需要单独的训练，导致专用模型的泛滥。
这种碎片化不仅增加了计算成本，而且从根本上限制了条件生成系统在实际应用中的灵活性。

当前C2I的方法主要侧重于提升特定条件类型的生成质量，而非解决条件兼容性这一根本问题。
像 ControlNet 和 T2I-Adapter 这样的方法需要为每种条件类型训练单独的模块，而像 Composer、UniControl 等方法需要修改架构以融入新的引导信号。
随着视觉条件的多样化，这种范式变得越来越不切实际，凸显了对能够无缝集成不同条件的统一框架的需求。

我们的出发点是，尽管视觉条件（例如草图、深度图、Canny图）看起来各不相同，
但它们本质上表示的是同一结构在不同视觉空间中的投影，共享可对齐的潜在特征。
不禁思考，是否可以通过有限的条件集估计这种潜在的一致性，从而将不同的条件联系起来，即使面对新的条件也能快速的归纳、融合、迁移。

于是我们提出了CRUCIBLE，它由一个支持热插拔的条件融合模块 (CAM)和一个Visual AR生成模型组成。
CRUCIBLE的核心是融合不同的条件，并找到共享的潜在信息元，实现对条件的统一， 
就像冶金师一样将不同金属熔于一炉，提纯去杂，最终锻造出具有卓越性能的合金。
所以实际上CRUCIBLE可以无缝衔接任何条件生成模型，是一个将条件处理与下游生成任务解耦的全新框架。
其中 CAM 负责将各种视觉条件转换为统一的目标表示，使现有的生成模型能够无缝地处理任何视觉条件，而无需重新训练和结构调整。
相较庞大的生成模型基座，CAM本身的微调效率极高，且对显存需求极低。最重要的是，CAM仅在条件间训练，不依赖基座模型的生成结果做反馈。

CRUCIBLE的核心创新在于向CAM引入基于正则化流的变分自编码器 (VAE) 瓶颈以及更稳定的概率密度损失。
相比传统的确定性方法：CAM依托信息瓶颈原理 [4]，描述了压缩和信息保留间的平衡范式。
通过将条件映射到分布而非点估计，最大化条件间的互信息的同时，最小化编码空间，从而生成不同条件间共享语义的浓缩表示。
其次，它强制约束结构化的潜在空间，无论原始条件如何，相似的语义概念都会被聚类，从而促进更好的泛化。
最后，基于正则化流和负熵惩罚的瓶颈层，可以最大程度上避免后验崩溃，生成有足够信息量的潜在表示。
正如 Rezende 和 Mohamed [7] 所证明的，基于流的后验分布可以建模更复杂的映射，并且具有易于处理的似然估计和采样。
这种能力对于我们的任务至关重要，因为条件空间之间的映射通常涉及多对多关系，纯粹的高斯分布难以拟合。

最后，为了将CRUCIBLE完善为一个完整的生成系统，我们还实现了一个更高效的条件控制VAR。
这得益于VAR的出现让我们看到了扩散模型并不是生成任务的唯一选择，并且AR模型让人惊讶的推理速度也进一步给生成模型的应用化带来希望。
更重要的是，现有C2I模型为了向庞大的拓展成本妥协，往往采用类似LoRA或者简单的向量注入来快速吸收视觉条件的信息。
但现在条件扩展的成本已经与基座模型解耦，我们希望探索更深度的条件融合策略，让生成模型将注意力集中于条件的控制而非区分，
迎合CRUCIBLE的初衷，解耦条件适配与条件控制。
通过将条件对齐重新表述为基于概率推理的分布迁移问题而非确定性的样本映射任务，
我们的工作建立了一种更具解释性和灵活性的方法来处理日益增长的视觉条件信号生态系统。条件、生成解耦的结构也为真正的通用生成任务提供思路。

综合来说，我们的贡献可以总结为以下几点：

1、提出了一个更符合通用条件生成模型期望的解耦架构CRUCIBLE。

2、提出了一个支持热插拔的条件对齐模块CAM。

3、提出了一个更深度融合条件的生成模型SCVAR。

4、在公开的数据集上取得了最新的生成效果。
# 3. 方法

CRUCIBLE由两部分组成：条件对齐模块（CAM）和条件生成模块（SCVAR）。
其中CAM接受常见的视觉条件作为输入，并将其输出为一个统一的目标条件（如：深度图）。 
SCVAR接受一个特定的条件输入，并在其监督下生成符合条件语义的自然图像。
再者，SCVAR区别于传统生成模型基于LoRA或简单特征注入的融合策略，SCVAR提出了基于ECFM的深度条件融合策略。
最后，CAM和SCVAR的训练是完全独立的。我们会分为两个部分，一一介绍两个模块的具体细节和理论依据。

## 3.1 CAM与条件一致性表征

CAM对齐条件的目的其实就是追求条件间的一致性表征，我们将这种一致性定义为：
给定一组视觉条件 $\mathcal{C} = \{c_1, c_2, ..., c_N\}$，
其中每个 $c_i \in \mathbb{R}^{H \times W \times C_i}$ 代表不同的条件类型（例如，深度图、草图、边缘图）。
我们的目标是学习一个统一的映射函数 $\mathcal{F}: c_i \rightarrow c_t$，
将任意条件 $c_i$ 转换为目标条件 $c_t \in \mathbb{R}^{H \times W \times C_t}$。
这使得下游生成模型能够专门针对 $c_t$ 进行操作，从而无需针对特定条件进行调整。

与以往学习确定性映射的方法不同，CAM通过一个基于分布的框架来建模这种转换，该框架能够捕捉条件空间之间固有的不确定性。
具体来说，CAM由四个关键组件组成：

1. **独立编码器** $E_i:\mathcal{C}_i\rightarrow\mathcal{F}_i$，用于提取针对每种条件类型的特征；
2. **共享编码器** $E_s:\mathcal{F}_i\rightarrow\mathcal{H}$，用于将条件特定特征映射到通用表示空间；
3. **带正则化流的 VAE 瓶颈**，用于将确定性表示转换为结构化的概率潜在空间 $\mathcal{Z}$；
4. **通用解码器** $D:\mathcal{Z}\rightarrow\mathcal{C}_t$，用于从潜在表征重构目标条件。

完整的流程可以表示为：
$$c_t = D(z),\quad\text{其中}\quad z\sim q_\phi(z|c_i)\quad\text{且}\quad q_\phi(z|c_i) = f_\theta(q_0(z_0|E_s(E_i(c_i))))$$

其中，$q_0(z_0|E_s(E_i(c_i)))$表示初始后验分布，$f_\theta$表示正则化流变换。

## 3.2 条件特定编码器和共享编码器

### 3.2.1 基于信息解耦的多条件一致编码

分离的独立编码器和共享编码器的设计，其实类似多任务学习，Caruana [20] 提出只要任务间存在统计相关性，那么模型就可以在多个输入源中，
共同学习目标表征，从而提高泛化性。 于是可以有这样的假设：每种条件类型都包含独特的信息和共享的语义内容。 
而CAM的目的就是尝试解耦这种共享信息。用信息论的语言描述就是将条件 $c_i$ 中的信息分解 [21] 为：
$$I(c_i; c_t) = I_{\text{unique}}(c_i; c_t) + I_{\text{shared}}(c_i; c_t; \{c_j\}_{j \neq i}) + I_{\cap}$$

其中 $I_{\text{unique}}$ 表示条件类型 $i$ 独有的信息，$I_{\text{shared}}$ 表示不同条件类型共有的信息'，$I_{\cap}$ 表示冗余的噪声。
而双阶段编码器的目的就是为了建模这种分解：

1. **条件特定编码器** ($E_i$) 提取特定类型的特征，捕获 $I_{\text{unique}}$
2. **共享编码器** ($E_s$) 提炼通用语义内容，重点关注 $I_{\text{shared}}$

首先，通过隔离特定条件的处理，我们可以防止“负迁移” [22]。因为在差异大的条件间进行迁移会有负面作用。形式上，独立解码器对于具有不同统计特性的条件
$c_i$ 和 $c_j$，有以下假设：
$$\nabla_{\theta_i} \mathcal{L}(c_i) \cdot \nabla_{\theta_j} \mathcal{L}(c_j) \approx 0$$

其中 $\theta_i$ 和 $\theta_j$ 是各自特定于条件的编码器的参数。所以，独立编码器可以用于吸收不同条件梯度间相对正交的部分，
缓解共享解码器的梯度震荡

最后，由共享编码器在相对靠近了的“任务”之间找到他们的最优归纳偏置 $\theta$ [23] ，促进“表征迁移”，
即：从已知条件类型中学习到的通用结构迁移到新的条件中。 对于新的条件类型 $c_{\text{new}}$，其理论误差界限为：
$$\mathbb{E}[\text{error}(c_{\text{new}})] \leq \mathbb{E}[\text{error}(c_{\text{seen}})] + d_{\mathcal{H}}(\mathcal{P}_{\text{seen}}, \mathcal{P}_{\text{new}})$$

其中 $d_{\mathcal{H}}$ 是共享表征空间中特征分布之间的差异。CAM通过一个共享编码器，最小化了这种差异，从而从不同条件中融合出高效的元特征。

### 3.2.2 实现

全部编码器 $E_i$ 都采用类似 U-Net 的卷积block作为基础结构。但相比较专属编码器，共享编码器 $E_s$ 需要面对更复杂的特征分布，所以我们引入了额外的
特征金字塔网络 (FPN) [9] 和跨尺度特征融合技术进一步处理这些特征。另外，所有的卷积block都加入了CBAM [8] 用于引导网络关注更有用的特征区域。
针对这些具体模块的实现细节，可以参考附录。

## 3.3 信息瓶颈与 基于流的VAE 框架

### 3.3.1 信息瓶颈原理

CAM的核心理论建立在信息瓶颈 (IB) 原理 [10] 之上，它提供了一个提取最小充分统计量的正式框架。在CAM的上下文中，IB 目标可以表述为：

$$\min_{p(z|c_i)} \; I(z; c_i) - \beta I(z; c_t)$$

该目标寻求一种表示 $z$，使其与输入条件 $I(z; c_i)$（压缩）的互信息最小化，同时与目标条件 $I(z; c_t)$（保留相关信息）的互信息最大化。
参数 $\beta$ 控制着这种权衡。

IB 原理特别适合处理多条件的融合与对齐问题，因为它可以自然地处理多对一映射的特性。多个源条件可以融合为一个最佳表示中，丢弃特定于条件的细节，
同时保留与目标条件相关的特征。

### 3.3.2 从信息瓶颈到变分自编码器

虽然信息瓶颈目标在理论上很优雅，但直接优化互信息项在计算上却很棘手。
参考Alemi 等人 [24] 的工作，我们尝试利用变分自编码器 (VAE) 作为信息瓶颈的一种实用近似。对于标准 VAE 目标函数：

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

他和 IB 目标函数的关系可以理解为，重构项近似于 $I(z; c_t)$，而 KL 散度项作为 $I(z; c_i)$ 的上界。

代入CAM中，我们将编码分布 $q_\phi(z|c_i)$ 建模为变分后验，令输入条件映射到潜在分布，而不是点估计。这种概率建模至关重要，原因如下：

1. 它自然地模拟了条件对齐中固有的不确定性，如：条件间的多对多映射。

2. 它强制构建一个结构化的潜在空间，聚集相似的语义概念，从而提高泛化能力。

3. 它提供了一种在压缩和保留相关结构之间取得平衡的原则性方法。

但由于传统VAE使用简单的高斯后验概率，使其无法建模复杂的条件关系。如果没有额外的约束，标准 VAE 的潜空间通常缺乏有效迁移到新条件类型所需的
额外非线性结构，所以表达能力有限，并且容易被强大的解码器忽略，导致后验坍塌 [25] 。这些限制对于条件对齐尤为重要，因为不同视觉条件
之间的映射逻辑十分复杂，涉及多对多的投影。

### 3.3.3 基于正则化流增强的变分后验

为了克服标准 VAE 的局限性，CAM引入了规范化流 [12] 来增强对更复杂分布的捕捉能力。 CAM首先将共享编码器输出 $h$ 映射到初始高斯后验分布的参数：

$$\mu = f_\mu(h), \quad \log\sigma = f_\sigma(h)$$
$$q(z|c_i) = \mathcal{N}(z; \mu, \sigma^2), \quad z_0 \sim q(z|c_i)$$

然后，通过一系列可逆的正则化流对这个初始分布进行变换：

$$z_k = f_k(z_{k-1}, h), \quad k \in \{1,...,K\}$$

其中每个 $f_k$ 都是以编码器输出 $h$ 为条件的可逆变换。变换后的变量 $z_K$ 的密度由变量变换公式给出：

$$\log q_K(z_K|c_i) = \log q_0(z_0|c_i) - \sum_{k=1}^{K} \log\left|\det\frac{\partial f_k}{\partial z_{k-1}}\right|$$

基于流的后验分布在Rezende 和 Mohamed [7] 的证明下，具备逼近任意复杂后验的能力，
有助于刻画不同条件类型间复杂的关系。Chen 等人 [26] 也指出，更具表达能力的后验可以更严格地约束信息瓶颈（IB）目标中的互信息，
从而实现更优的信息平衡。此外，与混合分布等方法不同，流方法保留了潜在空间的拓扑结构 [16]，确保语义上相近的样本在潜在空间中保持邻近，
有利于泛化到新条件。最后，流结构提供更稳定的梯度估计 [27] ，缓解 VAE 的不稳定问题。

### 3.3.4 实现

我们将每个流步骤实现为 ActNorm 层 [13] 和仿射耦合层 [14] 的组合，并针对计算效率进行了优化：

1. **ActNorm** 计算通道归一化以稳定激活值。

2. **仿射耦合** 将输入拆分为两部分，根据其中一部分对另一部分进行条件变换，然后重新组合它们：
$$z_{k,a}, z_{k,b} = \text{split}(z_{k-1})$$
$$h = \text{concat}(z_{k,a}, x)$$
$$\log s, t = \text{DS_block}_k(z_{k,1}, h)$$
$$z_{k,2} = z_{k-2} \odot \exp(\log s) + t$$
$$z_k = \text{concat}(z_{k,1}, z_{k,2}),\quad \log det=sum(\log s)$$

$DS\_block_k$代表 Depthwise Separable Convolution Block 可以在保持表达能力的同时兼顾计算效率，详情请参阅补充材料。

## 3.4 通用解码器

### 3.4.1 理论动机

CAM 使用一个在所有条件类型之间共享的通用解码器从$z_k$中恢复$c_t$。虽然编码过程必须处理输入条件的多样性，但解码过程始终针对相同的条件空间。 
从概率的角度来看，解码器在给定潜在表征的情况下，对目标条件的条件分布 $p_\theta(c_t|z)$ 进行建模。
由于$c_t$的分布不受输入条件类型的影响，因此共享解码器参数不仅更高效，
而且可以引导最优表征 $z$ 最小化重建 $c_t$ 所需的充分统计量的同时，最大化跨条件一致信息量。
这意味着给定一个最优潜在表征 $z^*$，条件独立性成立 [28] ：

$$p(c_t|z^*, c_i) = p(c_t|z^*)$$

换句话说，给定两个不同的条件类型 $c_i$ 和 $c_j$，以及相应的潜在变量 $z_i$ 和 $z_j$， 共享解码器强制执行：

$$\mathbb{E}[D(z_i) - D(z_j)]^2 \approx 0 \quad \text{当} \quad c_i \text{和 } c_j \text{表示同一场景时}$$

通用解码器就是一个强大的正则化器，强制不同条件在公共空间中对齐，最终收敛到一致的潜在表征，实现“表征对齐” [29]。

### 3.4.2 实现

CAM的通用解码器遵循标准的 U-Net 上采样架构，带有来自编码器的跳跃连接，并通过 CBAM 注意力模块进行增强：

$$g^l = \text{UpBlock}^l(g^{l+1}, s^l), \quad l \in \{L-1, L-2, ..., 0\}$$

其中 $g^L = z$ 是潜在表示，$s^l$ 表示来自编码器的跳跃连接，$g^0 = \hat{c}_t$ 是重构的目标条件。
每个 $\text{UpBlock}$ 都包含上采样、卷积和注意力操作。

## 3.5 训练

CAM将重建保真度、KL散度以及促进结构化潜在空间的正则化项相结合：

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{z} \mathcal{L}_{z}$$

### 3.5.1 重建损失

我们使用预测目标条件与真实目标条件之间的 L1 距离：

$$\mathcal{L}_{\text{recon}} = \|D(z) - c_t\|_1$$

与 L2 损失相比，L1 损失能够实现更清晰的重建。视觉条件的空间信息相对稀疏，但对类似边缘的结构化信息较为敏感。

### 3.5.2 基于流的 KL 散度

为了正则化潜在空间，我们利用流变换后的后验概率和标准高斯先验概率之间的 KL 散度：

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q_K(z|c_i) \| p(z))$$

这可以高效地计算如下：

$$\mathcal{L}_{\text{KL}} = D_{\text{KL}}(q_0(z_0|c_i) \| p(z_0)) - \mathbb{E}_{q_0(z_0|c_i)}\left[\sum_{k=1}^{K} \log\left|\det\frac{\partial f_k}{\partial z_{k-1}}\right|\right]$$

其中第一项是初始高斯后验概率的解析 KL 函数：

$$D_{\text{KL}}(q_0(z_0|c_i) \| p(z_0)) = \frac{1}{2}\sum_{j}\left(\mu_j^2 + \sigma_j^2 - 1 - \log\sigma_j^2\right)$$

### 3.5.3 直接潜在正则化

CAM之所以能在潜空间稳定训练的一个关键技巧就是对潜在变量 $z$ 添加了直接正则化：

$$\mathcal{L}_{z} = \frac{1}{2}\|z\|_2^2$$

这种直接约束有几个重要的好处：

1. **提高稳定性**：正如 Kumar 等人所证明的那样。 [17] 中，直接潜在变量正则化有助于防止后验崩溃，这是 VAE 中模型忽略潜在变量的常见故障模式。

2. **增强泛化**：通过限制潜在空间，我们提高了容量的利用效率，迫使模型发现更有意义且更可泛化的表示 [18]。

这种对潜在变量的直接正则化可以理解为一种近似推理的形式，能够更好地平衡后验灵活性。
相似的观点也在Cemgil 等人 [19] 的研究中提到，

## 3.6 可扩展条件视觉自回归模型 (SCVAR)

虽然 CAM 提供了统一的条件对齐框架，但完整的图像生成系统需要高效的条件生成组件。
为此，我们引入了 SCVAR，这是一种可扩展的条件视觉自回归模型，它基于自回归图像建模的最新进展，并结合了高效的条件集成策略。

### 3.6.1 用于图像生成的自回归建模

自回归 (AR) 模型已成为基于扩散的图像生成方法的一种引人注目的替代方案，它在不影响生成质量的情况下显著提高了推理速度 [30]。
自回归建模的核心原理是将高维数据的联合分布分解为条件分布的乘积：

$$p(x) = \prod_{i=1}^{n} p(x_i | x_{<i})$$

其中 $x = (x_1, x_2, ..., x_n)$ 表示数据序列，$x_{<i} = (x_1, x_2, ..., x_{i-1})$ 表示 $x_i$ 之前的所有元素。
在图像生成中，这通常涉及按光栅扫描顺序或某些预定义序列对像素进行建模。 然而，传统的像素级自回归模型存在两个主要局限性：
(1) 顺序生成速度极慢；(2) 一维序列化无法捕获像素间的空间依赖关系。

### 3.6.2 视觉自回归模型

视觉自回归模型 (VAR) 的最新研究 [31] 通过将图像生成重新表述为跨多个尺度的由粗到精的过程，解决了这些限制。
VAR 不是对单个像素进行建模，而是将图像 $x$ 分解为一系列多尺度表示：

$$x = \{x^1, x^2, ..., x^S\}$$

其中 $x^s \in \mathbb{R}^{H_s \times W_s \times C_s}$ 表示尺度 $s$ 下的图像表示，其中 $H_s = H/2^{S-s}$，$W_s = W/2^{S-s}$。
联合分布随后分解为：

$$p(x) = p(x^1) \prod_{s=2}^{S} p(x^s | x^{<s})$$

此公式具有以下几个数学优势：

1. **尺度内的并行性**：在每个尺度 $s$ 上，$x^s$ 的所有元素都可以并行生成，与像素级 AR 模型相比，推理时间显著缩短。

2. **自然的层次结构**：由粗到精的方法与图像的自然层次结构相一致，其中全局结构优先于局部细节。

3. **有界上下文大小**：上下文大小呈多项式增长而非指数增长，这使得长距离依赖关系建模更易于处理。

VAR 使用基于 Transformer 的架构实现了这种多尺度增强现实 (AR) 过程，该架构为每个尺度都设置了一个尺度预测头 $h_\theta^s$：

$$p_\theta(x^s | x^{<s}) = \text{softmax}(h_\theta^s(f^s))$$

其中 $f^s$ 表示尺度 $s$ 下的特征图，该特征图是通过 Transformer 主干网络处理所有先前尺度后得到的。

### 3.6.3 可扩展条件视觉增强现实 (SCVAR)

虽然 VAR 为无条件图像生成提供了一个高效的范式，但视觉条件控制下的生成过程仍然具有挑战性。
诸如简单的特征串联或全局条件嵌入等简单的方法无法充分利用视觉条件中的结构信息。
于是SCVAR 提出的高效条件融合模块 (ECFM) 扩展了 VAR，使其能够整合视觉条件。条件生成过程公式如下：

$$p(x|c) = p(x^1|c) \prod_{s=2}^{S} p(x^s | x^{<s}, c)$$

其中 $c \in \mathbb{R}^{H \times W \times C_c}$ 表示由 CAM 模块对齐后的目标条件（例如深度图、边缘图）。

### 3.6.4 高效条件融合模块 (ECFM)

SCVAR 的关键创新在于高效条件融合模块 (ECFM)，它解决了三个关键挑战：

1. **尺度对齐**：确保条件特征与每个尺度上的图像特征对齐
2. **有效融合**：在不中断 AR 建模过程的情况下融合条件信息
3. **计算效率**：平衡条件强度和推理速度

为了实现条件特征与图像特征在各个尺度上的精确对齐，我们将条件 $c$ 分解为与 VAR 主干模型匹配的多尺度表示：

$$c = \{c^1, c^2, ..., c^S\}$$

其中每个 $c^s \in \mathbb{R}^{H_s \times W_s \times C_c}$ 对应于尺度 $s$ 下的条件。此分解通过轻量级下采样网络 $D_c$ 实现：

$$c^s = D_c^s(c), \quad s \in \{1,2, ...,S\}$$

这种多尺度对齐确保生成过程的每个尺度都能以适当的分辨率接收条件指导，而条件也能根据具体的生成进度来提供恰到好处的引导强度。
但仅靠分辨率匹配无法处理复杂的条件细节。所以，ECFM 真正的核心是一种组合注意力机制，它通过两个连续的注意力操作将条件信息集成到主干特征中：

1. **交叉注意力**用于条件集成：
$$f_{\text{cross}}^s = \text{CrossAttn}(Q=f^s, K=c^s, V=c^s)$$

2. **自注意力**用于特征细化：
$$f_{\text{out}}^s = \text{SelfAttn}(Q=f_{\text{cross}}^s, K=f_{\text{cross}}^s, V=f_{\text{cross}}^s)$$

其中 $f^s$ 表示 $s$ 尺度上的主干特征。 具体来说，注意力操作可以表示为：

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

这种双重注意力方法允许模型首先通过交叉注意力整合相关的条件信息，然后通过自注意力细化和传播这些信息，从而创建更完整的表征。

注意力机制虽然能提供强大的全局处理能力，但也意味着庞大的参数开销。ECFM的高效就体现为，
在每个 Transformer 模块上应用交叉注意力计算成本高昂，而且很大程度上是冗余的。
我们观察到，条件信息通过自注意力和前馈层在网络中连续传递，信息强度随着距离递减，而相邻的block间的衰减并不严重。
于是，ECFM提出了一种间隔的条件注入策略，其中交叉注意力仅在整个网络中以固定的间隔应用。对于具有 $L$ 个模块的 Transformer，
我们在以下模块上应用 ECFM：

$$\{l | l \mod k = 0, l \in \{1,2,...,L\}\}$$

其中 $k$ 是跳跃间隔超参数。 具体来说，对于位置 $l$ 的 Transformer 模块 $B_l$，计算过程如下：

$$f_l = \begin{cases}
B_l(f_{l-1}) & \text{if } l \mod k \neq 0 \\
B_l(\text{ECFM}(f_{l-1}, c^s)) & \text{if } l \mod k = 0
\end{cases}$$

这种方法提供了一种在计算效率和条件强度之间取得平衡的原则性方法。
我们的实验表明，通过适当选择跳过间隔 $k$，SCVAR 可以实现与完整应用注意力机制相当的条件反射性能，
同时将条件反射组件的计算开销降低高达 $(k-1)/k \times 100\%$。

### 3.8.5 训练与推理

SCVAR 的训练目标是最大化训练数据的条件对数似然：

$$\mathcal{L}_{\text{SCVAR}} = -\mathbb{E}_{(x,c) \sim \mathcal{D}} \left[ \log p(x^1|c) + \sum_{s=2}^{S} \log p(x^s | x^{<s}, c) \right]$$

在推理过程中，我们以由粗到精的方式生成图像，并同时考虑先前生成的尺度和视觉条件：

1. 生成最粗尺度：$\hat{x}^1 \sim p(x^1|c)$
2. 对于每个后续尺度 $s \in \{2,...,S\}$：
- 生成以先前尺度为条件的尺度 $s$：$\hat{x}^s \sim p(x^s | x^{<s}, c)$

SCVAR 相较于现有的条件生成方法具有以下优势：

1. **速度**：与基于扩散的模型相比，多尺度自回归方法可以显著加快推理速度。

2. **高效的条件处理**：采用跳过条件处理策略的 ECFM 降低了计算开销，同时保持了强大的条件指导性。

3. **深度集成**：与 LoRA 或向量注入等轻量级自适应方法不同，ECFM 能够在生成过程的多个层面深度集成条件信息。

通过将视觉自回归建模的效率与我们新颖的条件方法相结合，SCVAR 完善了 CRUCIBLE 系统，为跨不同视觉条件的条件图像生成提供了强大而高效的框架。


[7] Rezende, D., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 1530-1538.

[8] Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). CBAM: Convolutional Block Attention Module. In Proceedings of the European Conference on Computer Vision (ECCV), 3-19.

[9] Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., & Belongie, S. (2017). Feature Pyramid Networks for Object Detection. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2117-2125.

[10] Tishby, N., Pereira, F. C., & Bialek, W. (2000). The Information Bottleneck Method. arXiv preprint physics/0004057.

[12] Rezende, D., & Mohamed, S. (2015). Variational Inference with Normalizing Flows. In Proceedings of the 32nd International Conference on Machine Learning (ICML), 1530-1538.

[13] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative Flow with Invertible 1x1 Convolutions. In Advances in Neural Information Processing Systems (NeurIPS), 10215-10224.

[14] Dinh, L., Sohl-Dickstein, J., & Bengio, S. (2017). Density Estimation Using Real NVP. In International Conference on Learning Representations (ICLR).

[16] Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., & Lakshminarayanan, B. (2021). Normalizing Flows for Probabilistic Modeling and Inference. Journal of Machine Learning Research, 22(57), 1-64.

[17] Kumar, A., Poole, B., & Murphy, K. (2020). Regularized Autoencoders via Relaxed Injective Probability Flow. In International Conference on Artificial Intelligence and Statistics (AISTATS), 4292-4301.

[18] Burgess, C. P., Higgins, I., Pal, A., Matthey, L., Watters, N., Desjardins, G., & Lerchner, A. (2018). Understanding Disentangling in β-VAE. arXiv preprint arXiv:1804.03599.

[19] Cemgil, T., Ghaisas, S., Goyal, P., & Dvijotham, K. (2020). The Autoencoding Variational Autoencoder. In Advances in Neural Information Processing Systems (NeurIPS), 33, 15970-15981.

[20] Caruana, R. (1997). Multitask Learning. Machine Learning, 28(1), 41-75.

[21] Williams, P. L., & Beer, R. D. (2010). Nonnegative Decomposition of Multivariate Information. arXiv preprint arXiv:1004.2515.

[22] Wang, Z., & Carbonell, J. (2018). Towards More Reliable Transfer Learning. In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD), 393-409.

[23] Baxter, J. (2000). A Model of Inductive Bias Learning. Journal of Artificial Intelligence Research, 12, 149-198.

[24] Alemi, A. A., Fischer, I., Dillon, J. V., & Murphy, K. (2017). Deep Variational Information Bottleneck. In International Conference on Learning Representations (ICLR).

[25] Razavi, A., van den Oord, A., & Vinyals, O. (2019). Generating Diverse High-Fidelity Images with VQ-VAE-2. In Advances in Neural Information Processing Systems (NeurIPS), 32, 14866-14876.

[26] Chen, X., Chen, R. T. Q., Metaxas, D. N., & Kingma, D. P. (2022). Stochastic Neural Network with Kronecker Flow. In International Conference on Learning Representations (ICLR).

[27] Huang, C.-W., Tan, S., Lacoste-Julien, S., & Courville, A. (2020). Improving Explorability in Variational Inference with Annealed Variational Objectives. In Advances in Neural Information Processing Systems (NeurIPS), 33, 9603-9613.

[28] Achille, A., & Soatto, S. (2018). Information Dropout: Learning Optimal Representations Through Noisy Computation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(12), 2897-2905.

[29] Wang, T., & Isola, P. (2020). Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. In International Conference on Machine Learning (ICML), 9929-9939.