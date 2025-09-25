import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborSampler

# 导入自定义模块
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="模型测试与指标评估")
    
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
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/best_model.ckpt",
                        help="模型检查点路径")
    parser.add_argument("--output_dir", type=str, default="./test_results/",
                        help="输出目录")
    
    # 测试相关参数
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小")
    
    return parser.parse_args()

def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_metric_comparison(metrics, metric_names, output_path):
    """绘制不同指标的对比图"""
    plt.figure(figsize=(10, 6))
    x = range(len(metric_names))
    plt.bar(x, metrics, color='royalblue')
    plt.xticks(x, metric_names)
    plt.ylim(0, 1.0)
    plt.ylabel('Score')
    plt.title('模型评估指标')
    
    # 在柱状图上方添加具体数值
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_loss_curve(losses, output_path):
    """绘制损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses, marker='o', linestyle='-', color='royalblue')
    plt.xlabel('批次')
    plt.ylabel('损失值')
    plt.title('测试损失曲线')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, data, args):
    """评估模型性能"""
    model.eval()
    
    # 准备测试数据
    print("正在准备测试数据...")
    test_idx = data.finetune_test_idx
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask[test_idx] = True
    
    # 过滤掉标签为-1的样本
    valid_mask = (data.y != -1) & test_mask
    test_idx_filtered = torch.nonzero(valid_mask).squeeze()
    
    # 使用ClusterLoader
    cluster_data = ClusterData(data, num_parts=100, recursive=False, save_dir=None)
    test_loader = ClusterLoader(cluster_data, batch_size=args.batch_size, shuffle=False)
    
    print(f"测试样本数量: {test_idx_filtered.size(0)}")
    
    # 收集所有预测和真实标签
    all_preds = []
    all_labels = []
    all_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 过滤当前批次中的测试样本
            batch_test_mask = torch.zeros(batch.num_nodes, dtype=torch.bool)
            # 将全局索引映射到批次内局部索引
            for global_idx in test_idx_filtered:
                local_indices = (batch.n_id == global_idx).nonzero()
                if local_indices.size(0) > 0:
                    batch_test_mask[local_indices[0]] = True
            
            # 如果当前批次没有测试样本，跳过
            if batch_test_mask.sum() == 0:
                continue
                
            # 前向传播
            batch = batch.to(model.device)
            result = model.test_step(batch)
            loss = result["loss"]
            logits = result["logits"]
            
            # 获取预测结果
            preds = torch.argmax(logits, dim=1)
            
            # 仅考虑测试集样本
            batch_labels = batch.y[batch_test_mask]
            batch_preds = preds[batch_test_mask]
            
            all_preds.extend(batch_preds.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            all_losses.append(loss.item())
    
    # 转换为numpy数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    avg_loss = np.mean(all_losses)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'loss': avg_loss,
        'all_losses': all_losses
    }
    
    return metrics

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
    
    # 评估模型
    print("正在评估模型...")
    metrics = evaluate_model(model, finetune_data, args)
    
    # 打印指标
    print("\n----- 模型评估结果 -----")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"平均损失: {metrics['loss']:.4f}")
    print("混淆矩阵:")
    print(metrics['confusion_matrix'])
    
    # 绘制并保存图表
    class_names = ['人类', '机器人', '水军']
    
    # 混淆矩阵
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(metrics['confusion_matrix'], class_names, cm_path)
    print(f"混淆矩阵已保存至: {cm_path}")
    
    # 指标对比图
    metrics_path = os.path.join(args.output_dir, 'metrics_comparison.png')
    metric_values = [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
    metric_names = ['准确率', '精确率', '召回率', 'F1分数']
    plot_metric_comparison(metric_values, metric_names, metrics_path)
    print(f"指标对比图已保存至: {metrics_path}")
    
    # 损失曲线
    loss_path = os.path.join(args.output_dir, 'loss_curve.png')
    plot_loss_curve(metrics['all_losses'], loss_path)
    print(f"损失曲线已保存至: {loss_path}")
    
    # 将指标保存到文本文件
    metrics_txt_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_txt_path, 'w') as f:
        f.write("----- 模型评估结果 -----\n")
        f.write(f"准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"精确率: {metrics['precision']:.4f}\n")
        f.write(f"召回率: {metrics['recall']:.4f}\n")
        f.write(f"F1分数: {metrics['f1']:.4f}\n")
        f.write(f"平均损失: {metrics['loss']:.4f}\n")
        f.write("混淆矩阵:\n")
        f.write(str(metrics['confusion_matrix']))
    
    print(f"所有指标已保存至: {metrics_txt_path}")
    print("\n测试完成!")

if __name__ == "__main__":
    main() 