import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import pytorch_lightning as pl
from torch_geometric.loader import ClusterData, ClusterLoader
import pandas as pd

# 导入自定义模块
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="模型特征重要性和性能分析")
    
    # 数据相关参数
    parser.add_argument("--dataset_path", type=str, default="./processed_data/",
                        help="数据集路径")
    parser.add_argument("--lst", action="store_true", default=True,
                        help="是否使用列表特征")
    parser.add_argument("--list_num", type=int, default=200000,
                        help="列表数量")
    parser.add_argument("--edge_types", type=str, default="ff",
                        help="使用的边类型")
    parser.add_argument("--llm_model", type=str, default="mistral",
                        help="使用的LLM模型名称")
    parser.add_argument("--llm_enhancement_channel", type=int, default=768,
                        help="LLM特征通道数")

    # 模型相关参数
    parser.add_argument("--pretext_task", type=str, default="contrastive",
                        help="预训练任务类型")
    parser.add_argument("--template", type=str, default="s",
                        help="模板类型")
    parser.add_argument("--checkpoint_path", type=str, default="G:\SeGA-LLM\SeGA-LLM\checkpoints\20250410-171222_finetuned_model.pt",
                        help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, default="./analysis_results/",
                        help="输出目录")
    
    # 分析相关参数
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小")
    parser.add_argument("--n_permutations", type=int, default=5,
                        help="特征重要性置换次数")
    
    return parser.parse_args()

def analyze_feature_importance(model, data, args):
    """分析特征重要性"""
    model.eval()
    
    # 准备测试数据
    print("正在准备测试数据...")
    test_idx = data.finetune_test_idx
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    # 过滤掉标签为-1的样本
    valid_mask = (data.y != -1) & test_mask
    test_idx_filtered = torch.nonzero(valid_mask).squeeze()
    
    # 提取测试样本的特征
    x_test = data.x[test_idx_filtered]
    y_test = data.y[test_idx_filtered]
    
    # 使用原始特征评估基准性能
    print("评估基准性能...")
    baseline_metrics = evaluate_baseline(model, data, test_idx_filtered)
    baseline_f1 = baseline_metrics['f1']
    print(f"基准F1分数: {baseline_f1:.4f}")
    
    # 特征分组
    feature_groups = {
        "用户属性特征": (0, 15),  # 假设的特征索引范围，需要根据实际调整
        "用户文本特征": (16, 200),
        "用户交互特征": (201, 300),
        "LLM增强特征": (301, 301 + args.llm_enhancement_channel)
    }
    
    # 计算特征重要性
    feature_importance = {}
    
    for group_name, (start_idx, end_idx) in feature_groups.items():
        print(f"分析 {group_name} 重要性...")
        
        # 执行多次置换以获得稳定的重要性评估
        group_importance_scores = []
        
        for i in range(args.n_permutations):
            print(f"  置换 {i+1}/{args.n_permutations}")
            
            # 创建一个特征置换的副本
            data_permuted = data.clone()
            
            # 随机打乱特定特征组
            permuted_features = data_permuted.x.clone()
            permuted_indices = torch.randperm(data.num_nodes)
            permuted_features[:, start_idx:end_idx] = permuted_features[permuted_indices][:, start_idx:end_idx]
            data_permuted.x = permuted_features
            
            # 评估置换后的性能
            permuted_metrics = evaluate_baseline(model, data_permuted, test_idx_filtered)
            permuted_f1 = permuted_metrics['f1']
            
            # 计算重要性分数（基线F1 - 置换后F1）
            importance = baseline_f1 - permuted_f1
            group_importance_scores.append(importance)
        
        # 计算平均重要性分数
        avg_importance = np.mean(group_importance_scores)
        feature_importance[group_name] = avg_importance
        print(f"  {group_name} 重要性分数: {avg_importance:.4f}")
    
    return feature_importance

def evaluate_baseline(model, data, test_idx):
    """评估基线性能"""
    model.eval()
    
    with torch.no_grad():
        # 创建一个小批次评估数据
        batch_data = data.clone()
        batch_data.batch = torch.zeros(batch_data.num_nodes, dtype=torch.long)
        batch_data = batch_data.to(model.device)
        
        # 前向传播
        result = model.test_step(batch_data)
        logits = result["logits"]
        
        # 获取预测结果
        preds = torch.argmax(logits, dim=1)
        
        # 过滤测试样本
        test_mask = torch.zeros(batch_data.num_nodes, dtype=torch.bool)
        for idx in test_idx:
            test_mask[idx] = True
        
        y_true = batch_data.y[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro'),
            'recall': recall_score(y_true, y_pred, average='macro'),
            'f1': f1_score(y_true, y_pred, average='macro')
        }
        
        return metrics

def analyze_performance_by_attribute(model, data, args):
    """分析不同属性下的模型性能"""
    model.eval()
    
    # 准备测试数据
    print("正在准备测试数据...")
    test_idx = data.finetune_test_idx
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    # 过滤掉标签为-1的样本
    valid_mask = (data.y != -1) & test_mask
    test_idx_filtered = torch.nonzero(valid_mask).squeeze()
    
    # 提取测试样本的特征和标签
    x_test = data.x[test_idx_filtered]
    y_test = data.y[test_idx_filtered]
    
    # 获取预测
    with torch.no_grad():
        # 创建一个批次评估数据
        batch_data = data.clone()
        batch_data.batch = torch.zeros(batch_data.num_nodes, dtype=torch.long)
        batch_data = batch_data.to(model.device)
        
        # 前向传播
        result = model.test_step(batch_data)
        logits = result["logits"]
        
        # 获取预测结果
        preds = torch.argmax(logits, dim=1)
        
        # 过滤测试样本
        test_mask = torch.zeros(batch_data.num_nodes, dtype=torch.bool)
        for idx in test_idx_filtered:
            test_mask[idx] = True
        
        y_true = batch_data.y[test_mask].cpu().numpy()
        y_pred = preds[test_mask].cpu().numpy()
    
    # 假设特征0是用户活跃天数，特征1是关注者数量
    # 这里需要根据实际特征索引调整
    activity_days = x_test[:, 0].cpu().numpy()  # 活跃天数
    followers_count = x_test[:, 1].cpu().numpy()  # 关注者数量
    
    # 按关注者数量分组
    followers_bins = [0, 10, 100, 1000, np.inf]
    followers_labels = ['0-10', '11-100', '101-1000', '1000+']
    followers_groups = pd.cut(followers_count, bins=followers_bins, labels=followers_labels)
    
    # 按活跃天数分组
    activity_bins = [0, 30, 90, 365, np.inf]
    activity_labels = ['0-30天', '31-90天', '91-365天', '1年以上']
    activity_groups = pd.cut(activity_days, bins=activity_bins, labels=activity_labels)
    
    # 分组计算性能指标
    performance_by_followers = {}
    for group in followers_labels:
        group_mask = followers_groups == group
        if np.sum(group_mask) == 0:
            continue
            
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        performance_by_followers[group] = {
            'count': np.sum(group_mask),
            'accuracy': accuracy_score(group_y_true, group_y_pred),
            'precision': precision_score(group_y_true, group_y_pred, average='macro'),
            'recall': recall_score(group_y_true, group_y_pred, average='macro'),
            'f1': f1_score(group_y_true, group_y_pred, average='macro')
        }
    
    performance_by_activity = {}
    for group in activity_labels:
        group_mask = activity_groups == group
        if np.sum(group_mask) == 0:
            continue
            
        group_y_true = y_true[group_mask]
        group_y_pred = y_pred[group_mask]
        
        performance_by_activity[group] = {
            'count': np.sum(group_mask),
            'accuracy': accuracy_score(group_y_true, group_y_pred),
            'precision': precision_score(group_y_true, group_y_pred, average='macro'),
            'recall': recall_score(group_y_true, group_y_pred, average='macro'),
            'f1': f1_score(group_y_true, group_y_pred, average='macro')
        }
    
    return {
        'followers': performance_by_followers,
        'activity': performance_by_activity
    }

def plot_feature_importance(feature_importance, output_path):
    """绘制特征重要性图表"""
    plt.figure(figsize=(12, 6))
    
    # 按重要性排序
    sorted_importance = {k: v for k, v in sorted(feature_importance.items(), 
                                               key=lambda item: item[1], reverse=True)}
    
    features = list(sorted_importance.keys())
    importance_scores = list(sorted_importance.values())
    
    # 创建横向条形图
    bars = plt.barh(features, importance_scores, color='royalblue')
    
    # 添加数值标签
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{importance_scores[i]:.4f}', va='center')
    
    plt.xlabel('重要性分数 (基线F1 - 置换F1)')
    plt.ylabel('特征组')
    plt.title('特征组重要性分析')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_performance_by_attribute(performance_data, attribute_name, output_path):
    """绘制按属性分组的性能指标"""
    plt.figure(figsize=(14, 8))
    
    # 提取数据
    groups = list(performance_data.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    counts = [performance_data[group]['count'] for group in groups]
    
    # 设置位置
    x = np.arange(len(groups))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]
    
    # 绘制每个指标的条形图
    for i, metric in enumerate(metrics):
        values = [performance_data[group][metric] for group in groups]
        plt.bar(x + offsets[i], values, width, label=metric.capitalize())
    
    # 添加样本数量标签
    for i, count in enumerate(counts):
        plt.text(x[i], 0.05, f'n={count}', ha='center', rotation=90, alpha=0.7)
    
    # 添加标签和图例
    plt.xlabel(f'{attribute_name}分组')
    plt.ylabel('性能分数')
    plt.title(f'按{attribute_name}分组的模型性能')
    plt.xticks(x, groups)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据
    print("正在加载数据...")
    pretrain_data, finetune_data = load_data_llm(args)
    
    # 加载模型
    print(f"正在加载模型检查点: {args.checkpoint_path}")
    model = SeGA_LLM.load_from_checkpoint(args.checkpoint_path)
    model.eval()
    
    # 分析特征重要性
    print("正在分析特征重要性...")
    feature_importance = analyze_feature_importance(model, finetune_data, args)
    
    # 绘制特征重要性图表
    importance_path = os.path.join(args.output_dir, 'feature_importance.png')
    plot_feature_importance(feature_importance, importance_path)
    print(f"特征重要性图表已保存至: {importance_path}")
    
    # 分析不同属性下的性能
    print("正在分析不同属性下的性能...")
    performance_by_attribute = analyze_performance_by_attribute(model, finetune_data, args)
    
    # 绘制按属性分组的性能图表
    followers_path = os.path.join(args.output_dir, 'performance_by_followers.png')
    plot_performance_by_attribute(performance_by_attribute['followers'], '关注者数量', followers_path)
    print(f"按关注者数量的性能图表已保存至: {followers_path}")
    
    activity_path = os.path.join(args.output_dir, 'performance_by_activity.png')
    plot_performance_by_attribute(performance_by_attribute['activity'], '账户活跃天数', activity_path)
    print(f"按活跃天数的性能图表已保存至: {activity_path}")
    
    # 保存分析结果到文本文件
    analysis_txt_path = os.path.join(args.output_dir, 'analysis_results.txt')
    with open(analysis_txt_path, 'w') as f:
        f.write("----- 特征重要性分析 -----\n")
        for feature, importance in feature_importance.items():
            f.write(f"{feature}: {importance:.4f}\n")
        
        f.write("\n----- 按关注者数量分组的性能 -----\n")
        for group, metrics in performance_by_attribute['followers'].items():
            f.write(f"{group} (样本数: {metrics['count']}):\n")
            f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"  精确率: {metrics['precision']:.4f}\n")
            f.write(f"  召回率: {metrics['recall']:.4f}\n")
            f.write(f"  F1分数: {metrics['f1']:.4f}\n")
        
        f.write("\n----- 按账户活跃天数分组的性能 -----\n")
        for group, metrics in performance_by_attribute['activity'].items():
            f.write(f"{group} (样本数: {metrics['count']}):\n")
            f.write(f"  准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"  精确率: {metrics['precision']:.4f}\n")
            f.write(f"  召回率: {metrics['recall']:.4f}\n")
            f.write(f"  F1分数: {metrics['f1']:.4f}\n")
    
    print(f"分析结果已保存至: {analysis_txt_path}")
    print("\n分析完成!")

if __name__ == "__main__":
    main() 