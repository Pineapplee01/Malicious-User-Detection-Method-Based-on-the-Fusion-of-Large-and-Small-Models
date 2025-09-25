# SeGA-LLM 模型评估工具

本目录包含了一系列用于评估 SeGA-LLM 模型性能的测试工具。这些工具可以生成各种性能指标和可视化图表，帮助分析模型的有效性和特征重要性。

## 主要功能

1. **基础指标评估** (`model_test.py`)
   - 准确率、精确率、召回率、F1分数
   - 混淆矩阵
   - 测试损失

2. **ROC和PR曲线分析** (`model_test_curves.py`)
   - 每个类别的ROC曲线和AUC值
   - 精确率-召回率曲线和平均精度(AP)值

3. **特征和属性分析** (`model_analysis.py`)
   - 特征重要性评估
   - 按用户属性分组的性能分析

4. **综合报告生成** (`run_tests.py`)
   - 运行所有测试并生成PDF格式的完整报告

## 使用方法

### 准备工作

确保已安装所有必要的依赖：

```bash
pip install torch pytorch-lightning matplotlib seaborn scikit-learn pandas torch-geometric
```

### 运行单个测试脚本

1. **基础指标评估**：

```bash
python model_test.py --checkpoint_path ./checkpoints/best_model.ckpt --dataset_path ./processed_data/ --output_dir ./test_results/
```

2. **ROC和PR曲线分析**：

```bash
python model_test_curves.py --checkpoint_path ./checkpoints/best_model.ckpt --dataset_path ./processed_data/ --output_dir ./test_results/
```

3. **特征和属性分析**：

```bash
python model_analysis.py --checkpoint_path ./checkpoints/best_model.ckpt --dataset_path ./processed_data/ --output_dir ./analysis_results/ --n_permutations 5
```

### 生成综合报告

运行一体化测试脚本，自动执行所有测试并生成综合PDF报告：

```bash
python run_tests.py --checkpoint_path ./checkpoints/best_model.ckpt --dataset_path ./processed_data/ --output_dir ./test_reports/
```

## 参数说明

所有脚本都支持以下通用参数：

| 参数                    | 描述                        | 默认值                   |
|------------------------|-----------------------------|--------------------------|
| --dataset_path         | 数据集路径                   | ./processed_data/        |
| --lst                  | 是否使用列表特征              | True                     |
| --list_num             | 列表数量                     | 200000                   |
| --edge_types           | 使用的边类型                 | ff                       |
| --llm_model            | 使用的LLM模型名称            | mistral                  |
| --llm_enhancement_channel | LLM特征通道数             | 768                      |
| --pretext_task         | 预训练任务类型               | contrastive              |
| --template             | 模板类型                     | s                        |
| --checkpoint_path      | 模型检查点路径               | ./checkpoints/best_model.ckpt |
| --output_dir           | 输出目录                     | 根据脚本不同有不同默认值    |
| --batch_size           | 批次大小                     | 128                      |

特征分析脚本特有参数：

| 参数                    | 描述                        | 默认值                   |
|------------------------|-----------------------------|--------------------------|
| --n_permutations       | 特征重要性置换次数           | 5                        |

## 输出说明

每个测试脚本都会在指定的输出目录中生成结果：

1. **基础指标评估**：
   - `metrics.txt` - 包含所有基础指标的文本文件
   - `confusion_matrix.png` - 混淆矩阵可视化
   - `metrics_comparison.png` - 各指标对比图
   - `loss_curve.png` - 测试损失曲线

2. **ROC和PR曲线分析**：
   - `roc_curve.png` - ROC曲线图
   - `pr_curve.png` - 精确率-召回率曲线图
   - `curves_metrics.txt` - 包含AUC和AP值的文本文件

3. **特征和属性分析**：
   - `feature_importance.png` - 特征重要性图
   - `performance_by_followers.png` - 按关注者数量分组的性能图
   - `performance_by_activity.png` - 按账户活跃天数分组的性能图
   - `analysis_results.txt` - 包含所有分析结果的文本文件

4. **综合报告**：
   - 一个包含所有结果的时间戳目录
   - 一个PDF格式的完整报告，包含所有图表和文本结果

## 高级用法

### 自定义特征分析

在`model_analysis.py`脚本中，您可以修改特征组的定义来适应您的特定模型：

```python
feature_groups = {
    "用户属性特征": (0, 15),      # 修改这些索引范围
    "用户文本特征": (16, 200),    # 以匹配您的特征维度
    "用户交互特征": (201, 300),
    "LLM增强特征": (301, 301 + args.llm_enhancement_channel)
}
```

### 调整图表样式

所有可视化函数都使用matplotlib和seaborn，您可以根据需要调整图表的样式、颜色和格式。 