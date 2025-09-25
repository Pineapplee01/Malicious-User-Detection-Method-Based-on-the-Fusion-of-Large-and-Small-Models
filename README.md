# SeGA-LLM: LLM增强的异常用户检测

本项目结合了SeGA (AAAI 2024) 的偏好感知自对比学习方法与大型语言模型 (LLMs) 技术，用于提高Twitter上的异常用户检测性能。这一实现主要改进了文本处理部分，利用LLM的强大语言理解能力来增强用户描述和推文的特征表示。

## 主要特点

- 集成了SeGA的图结构学习与LLM的文本分析能力
- 使用偏好感知自对比学习进行预训练
- 利用LLM生成丰富的语义表示，提升对文本数据的理解
- 支持多种LLM模型，包括Mistral、LLaMA-2和ChatGPT
- 保留了SeGA的原始架构，只在文本特征处理部分进行增强

## 环境要求

- Python 3.8+
- PyTorch 1.10+
- PyTorch Geometric
- Transformers
- CUDA支持 (GPU加速)

## 文件结构

- `SeGA_LLM.py`: LLM增强的SeGA模型实现
- `llm_feature_extractor.py`: 使用LLM提取增强特征的工具
- `preprocess_llm.py`: 预处理脚本，生成LLM增强的特征
- `data_loader_llm.py`: 支持LLM增强特征的数据加载器
- `main_llm.py`: 运行LLM增强版SeGA的主程序
- `globals.py`: 全局变量存储

## 使用步骤

### 1. 预处理：生成LLM增强特征

首先运行预处理脚本，使用LLM对用户和列表描述及推文进行分析，并生成增强特征：

```bash
python preprocess_llm.py --llm_model mistral --input_path /path/to/data --output_path ./processed_data --device 0
```

主要参数：
- `--llm_model`: 选择LLM模型，支持 'mistral', 'llama2_7b', 'llama2_13b', 'llama2_70b', 'chatgpt'
- `--input_path`: 输入数据路径
- `--output_path`: 输出处理后数据的路径
- `--device`: GPU设备ID

### 2. 训练与评估LLM增强的SeGA模型

运行主程序进行模型训练和评估：

```bash
python main_llm.py --llm_model mistral --dataset_path ./processed_data --pretrain --lst
```

主要参数：
- `--llm_model`: 使用的LLM模型类型
- `--dataset_path`: 处理后数据的路径
- `--pretrain`: 启用预训练阶段
- `--lst`: 使用列表节点
- `--output_dir`: 输出目录，默认为 './output/'

更多参数详见 `main_llm.py` 中的 `build_args` 函数。

## 与原始SeGA的区别

1. 文本特征处理：
   - 原始SeGA使用RoBERTa-base直接提取用户描述和推文的特征
   - SeGA-LLM先使用LLM分析文本内容，然后提取这些分析结果的特征，捕获更复杂的语义模式

2. 特征维度：
   - 原始SeGA的文本特征维度为768 (RoBERTa)
   - SeGA-LLM包含原始特征(768)、LLM描述分析特征(768)和LLM推文分析特征(768)，共2304维

3. 实现变化：
   - 新增了特征抽取管道
   - 修改了特征融合方式
   - 支持多种LLM模型选择

## 参考

- SeGA: [Preference-Aware Self-Contrastive Learning with Prompts for Anomalous User Detection on Twitter](https://arxiv.org/abs/2312.11553)
- Botsay: [What Does the Bot Say? Opportunities and Risks of Large Language Models in Social Media Bot Detection](https://arxiv.org/abs/2402.00371) 