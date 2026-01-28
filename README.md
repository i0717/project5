# 多模态情感分类实验（实验五）

本项目实现了基于文本（BERT）和图像（ResNet50）的多模态情感分类模型，针对 positive/neutral/negative 三分类任务，采用 Early Fusion 策略进行模态融合，并完成了完整的消融实验和数据增强优化。

## 项目概述
- **任务类型**：多模态情感三分类（positive/neutral/negative）
- **模态融合方式**：Early Fusion（文本特征+图像特征+元素级相乘）
- **核心模型**：
  - 文本编码器：BERT-base-uncased
  - 图像编码器：ResNet50
- **最佳性能**：验证集准确率 72.88%，加权F1 70.56%
- **数据增强**：文本清洗、同义词替换、图像随机裁剪/翻转/色彩增强

## 目录
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [模型下载](#模型下载)
- [快速开始](#快速开始)
- [实验结果](#实验结果)
- [代码说明](#代码说明)
- [参考资源](#参考资源)
- [常见问题](#常见问题)

## 项目结构
```
project5/
├── bert-base-uncased/          # 本地BERT模型文件（需自行下载）
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer_config.json
│   └── vocab.txt
├── data/                       # 数据集文件夹（包含txt文本和jpg图片）
│   ├── {guid}.txt              # 文本数据（按guid命名）
│   └── {guid}.jpg              # 图像数据（按guid命名）
├── train.txt                   # 训练集标签文件（格式：guid,tag）
├── test_without_label.txt      # 测试集文件（预测后会覆盖标签列）
├── project5.ipynb              # 核心代码文件（所有实验逻辑）
├── requirements.txt            # 环境依赖文件
├── best_early_fusion_with_exp2_aug.pth  # 最佳模型权重
├── early_fusion_exp2_aug_results.txt    # 主实验结果
├── early_fusion_ablation_results.txt    # 消融实验结果
└── README.md                   # 项目说明文档
```

## 环境配置

### 1. 依赖安装
创建并激活虚拟环境后，执行以下命令安装依赖：
```bash
pip install -r requirements.txt
```

### 2. requirements.txt 内容
```
python>=3.8
numpy==1.24.3
pandas==1.5.3
matplotlib==3.7.2
Pillow==10.0.0
tqdm==4.65.0
scikit-learn==1.3.0

# PyTorch核心
torch==2.0.1
torchvision==0.15.2

# 自然语言处理
transformers==4.30.2
nltk==3.8.1

# Jupyter开发环境
jupyter==1.0.0
ipykernel==6.25.0
ipywidgets==8.0.7
notebook==6.5.4

# 数据可视化
seaborn==0.12.2

# 性能优化
accelerate==0.20.3
```

### 3. 额外配置
- 下载 NLTK 资源（运行代码时自动下载，首次运行需联网）：
  ```python
  import nltk
  nltk.download('wordnet')
  ```
- 确保本地有 CUDA 环境（推荐 CUDA 11.8），若无 GPU 可使用 CPU 运行（速度较慢）



## 模型下载

由于BERT模型文件较大（~440MB），需要手动下载：

### 方式1：使用huggingface-cli下载（推荐）
```bash
# 安装huggingface-hub
pip install huggingface-hub

# 下载BERT模型到项目根目录
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='bert-base-uncased', local_dir='./bert-base-uncased')"
```

### 方式2：使用transformers库自动下载（首次运行代码时会自动下载）
代码配置为离线模式，首次运行时会自动下载模型到本地缓存目录。如需指定下载位置，请运行：
```python
from transformers import BertModel, BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.save_pretrained('./bert-base-uncased')
tokenizer.save_pretrained('./bert-base-uncased')
```

### 方式3：从官方GitHub下载
1. 访问 https://huggingface.co/bert-base-uncased/tree/main
2. 下载以下文件到 `bert-base-uncased/` 目录：
   - `config.json`
   - `pytorch_model.bin`
   - `tokenizer_config.json`
   - `vocab.txt`

### 下载后确认文件结构
```
bert-base-uncased/
├── config.json
├── pytorch_model.bin        # 核心权重文件，~440MB
├── tokenizer_config.json
└── vocab.txt
```


## 快速开始

### 1. 数据准备
- 将数据集解压到 `project5/` 目录下，确保 `data/` 文件夹包含所有文本和图像文件
- 确认 `train.txt` 和 `test_without_label.txt` 文件路径正确
- 本地 BERT 模型文件夹 `bert-base-uncased/` 需放在项目根目录

### 2. 代码执行流程
所有实验逻辑均在 `project5.ipynb` 中，按以下步骤执行：

#### 步骤1：配置路径
修改代码中 `Config` 类的 `base_path` 为你的项目根目录：
```python
class Config:
    base_path = r"你的项目根目录路径"  # 例如：r"D:\大学作业\当代人工智能\实验五\project5"
    # 其他配置保持不变
```

#### 步骤2：运行主实验
执行 notebook 中「最终模型」部分（倒数第二个单元格代码）：
- 加载训练/测试数据
- 划分训练集/验证集（8:2分层抽样）
- 应用数据增强（文本清洗+同义词替换+图像增强）
- 训练 Early Fusion 模型
- 预测测试集并覆盖 `test_without_label.txt`

#### 步骤3：运行消融实验
执行 notebook 中「消融实验」部分（倒数第一个单元格代码）：
- 加载训练好的最佳模型
- 分别测试仅文本、仅图像、完整多模态输入的性能
- 生成消融实验结果和可视化图表

#### 步骤4：查看结果
- 主实验结果保存在 `early_fusion_exp2_aug_results.txt`
- 消融实验结果保存在 `early_fusion_ablation_results.txt`
- 测试集预测结果直接覆盖在 `test_without_label.txt` 中

### 3. 关键参数说明
| 参数 | 取值 | 说明 |
|------|------|------|
| batch_size | 16 | 训练批次大小 |
| learning_rate | 1e-5 | 学习率 |
| hidden_dim | 512 | 融合层隐藏维度 |
| dropout | 0.3 | Dropout 概率 |
| num_epochs | 10 | 训练轮数 |
| test_size | 0.2 | 验证集比例 |

## 实验结果

### 1. 主实验结果
| 模型 | 验证集准确率 | 加权F1 | 最佳验证准确率 |
|------|--------------|--------|----------------|
| Early Fusion（带数据增强） | 72.88% | 70.56% | 72.88% |

### 2. 测试集预测分布
- positive: 345 (67.5%)
- neutral: 20 (3.9%)
- negative: 146 (28.6%)

### 3. 消融实验结果
| 输入方式 | 准确率 | 加权F1 | 相比多模态提升 |
|----------|--------|--------|----------------|
| 仅文本 | 64.50% | 65.19% | -8.38% |
| 仅图像 | 63.38% | 57.49% | -9.50% |
| 多模态 | 72.88% | 70.56% | - |

## 代码说明

### 核心模块
1. **数据预处理**：
   - `clean_text()`：文本清洗（移除URL、特殊字符、统一小写）
   - `synonym_replacement()`：情感同义词替换（数据增强）
   - 图像增强：随机裁剪、翻转、色彩抖动等

2. **数据集类**：
   - `MultimodalDataset`：支持文本/图像加载、数据增强、标签映射

3. **模型架构**：
   - `EarlyFusionModel`：BERT+ResNet50+Early Fusion融合策略
   - 文本/图像编码器部分冻结，仅微调最后几层

4. **训练器类**：
   - `Trainer`：包含训练、验证、模型保存逻辑
   - 支持早停、梯度裁剪、混合精度训练

5. **消融实验**：
   - `evaluate_text_only_early()`：仅文本输入评估
   - `evaluate_image_only_early()`：仅图像输入评估
   - `evaluate_full_multimodal_early()`：完整多模态评估

## 参考资源
1. **模型参考**：
   - BERT: https://arxiv.org/abs/1810.04805
   - ResNet: https://arxiv.org/abs/1512.03385
   - Early Fusion: https://arxiv.org/abs/1909.02950

2. **代码参考**：
   - Hugging Face Transformers: https://github.com/huggingface/transformers
   - PyTorch TorchVision: https://github.com/pytorch/vision

3. **数据集**：
   - 匿名多模态情感分类数据集（实验五专用）

## 注意事项
1. 首次运行需下载预训练模型权重，确保网络畅通
2. 训练过程中会生成模型权重文件（约1.3GB），确保磁盘空间充足
3. 若使用CPU训练，建议减小batch_size至8，训练时间会显著增加
4. 测试集预测结果会直接覆盖 `test_without_label.txt`，建议提前备份原始文件

## 常见问题

### Q1: BERT模型下载失败
- **问题**：网络连接问题导致BERT模型下载失败
- **解决方案**：
  1. 使用VPN或切换网络环境
  2. 手动从官方GitHub下载（方式3）
  3. 使用镜像源：
  ```python
  from transformers import BertModel, BertTokenizer
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='./cache')
  model = BertModel.from_pretrained('bert-base-uncased', cache_dir='./cache')
  ```

### Q2: 内存/显存不足
- **问题**：训练时出现CUDA out of memory错误
- **解决方案**：
  1. 减小batch_size（如从16改为8）
  2. 启用梯度累积
  3. 使用CPU模式训练（速度会变慢）

### Q3: 本地模型路径错误
- **问题**：找不到本地BERT模型文件
- **解决方案**：
  1. 确认 `bert-base-uncased/` 文件夹在项目根目录
  2. 确认文件夹内包含所有4个必需文件
  3. 修改代码中的模型路径：
  ```python
  text_model = os.path.join(base_path, "bert-base-uncased")
  ```


