# sarcasm_detection# 
多模态讽刺检测项目

## 项目简介
本项目实现了一个基于CLIP的多模态讽刺检测模型，通过创新的特征处理模块和交互机制，有效地识别文本和图像中的讽刺内容。

## 模型架构

### 核心组件

1. **CLIP基础模型**
   - 使用预训练的CLIP模型作为特征提取器
   - 分别处理文本和图像输入，提取多模态特征

2. **SFDA (Scaled Feature Decomposition and Aggregation)**
   - 改进的图像特征处理模块
   - 通过特征分解和聚合减少冗余，增强特征唯一性
   - 包含通道注意力和空间注意力机制

3. **DMI (Dynamic Adaptive Interaction)**
   - 动态自适应交互模块
   - 实现文本和图像特征的高效交互
   - 包含门控机制和自适应缩放因子

4. **损失函数设计**
   - 结合分类损失和Jensen-Shannon散度
   - 动态调整正则化权重

## 环境要求

- Python 3.6+
- PyTorch
- Transformers库
- CLIP预训练模型

## 主要文件说明

- `model.py`: 包含模型的核心实现
  - `MV_CLIP_new`: 主模型类
  - `SFDABlock`: 特征处理模块
  - `DynamicAdaptiveInteractionBlock`: 交互模块

- `main.py`: 训练和评估脚本

## 使用说明

1. 环境配置
   ```bash
   pip install torch transformers
   ```

2. 准备CLIP模型
   - 下载CLIP预训练模型
   - 更新模型路径配置

3. 运行训练
   ```bash
   python main.py
   ```

## 模型参数

主要可配置参数包括：
- `text_size`: 文本特征维度
- `image_size`: 图像特征维度
- `dropout_rate`: Dropout比率
- `label_number`: 分类标签数量

## 创新点

1. SFDA模块
   - 通过特征分解减少冗余
   - 多层注意力机制增强特征表示

2. DMI模块
   - 动态自适应的特征交互
   - 门控机制和自适应缩放

3. 损失函数设计
   - 动态调整的正则化项
   - 模态间一致性约束
