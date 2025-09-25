import argparse
import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve, 
    balanced_accuracy_score
)
from torch_geometric.loader import NeighborLoader
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import logging
import warnings
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import torch_geometric as pyg

# 导入自定义模块
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='SeGA-LLM 评估')
    parser.add_argument("--dataset_path", type=str, default="./processed_data",
                        help="处理后数据的路径")
    parser.add_argument('--model_path', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='评估结果输出目录')
    parser.add_argument('--dim', type=int, default=16,
                        help='嵌入维度')
    parser.add_argument('--edge_types', type=str, default="ff",
                        help='边类型')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批处理大小')
    parser.add_argument('--cuda', type=int, default=0,
                        help='GPU设备ID')
    parser.add_argument('--encoder', type=str, default='SimpleGCN',
                        help='编码器类型')
    parser.add_argument('--prompt_encoder', type=str, default='roberta',
                        help='提示编码器类型')
    parser.add_argument('--llm_model', type=str, default='mistral',
                        help='LLM模型类型')
    parser.add_argument('--llm_enhancement_channel', type=int, default=103,
                        help='LLM增强通道维度')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='温度参数') 
    parser.add_argument('--pretext_task', type=str, default='contrastive',
                        help='前置任务')
    parser.add_argument('--lst', action='store_true', default=True,
                        help='是否使用列表特征')
    parser.add_argument('--list_num', type=int, default=200000,
                        help='列表数量')
    parser.add_argument('--simplified', action='store_true', default=True,
                        help='使用简化模型')
    parser.add_argument('--split_ratio', type=str, default="0.8,0.05,0.15",
                        help='训练:验证:测试的比例')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    # 其他必要参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--epoch', type=int, default=1, help='训练轮数')
    parser.add_argument('--wd', type=float, default=0.0001, help='权重衰减')
    parser.add_argument('--sample', action='store_true', default=False, help='是否采样邻居')
    parser.add_argument('--pretrain', action='store_true', default=False, help='是否预训练')
    parser.add_argument('--template', type=str, default='s', help='模板类型')
    
    return parser.parse_args()

class SimpleBatchLoader(Dataset):
    def __init__(self, data, batch_size=32):
        self.data = data
        self.batch_size = batch_size
        self.length = len(data['user_node_masks'])

    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = min(start + self.batch_size, self.length)
        
        batch = {}
        for key, value in self.data.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value[start:end]
            elif isinstance(value, list):
                batch[key] = value[start:end]
            elif isinstance(value, dict):
                batch[key] = {k: v[start:end] if isinstance(v, torch.Tensor) or isinstance(v, list) else v for k, v in value.items()}
            else:
                batch[key] = value
        
        return batch

def create_data_loader(data, batch_size, indices, use_neighbor_loader=True):
    """创建数据加载器"""
    if use_neighbor_loader:
        try:
            return NeighborLoader(
                data, 
                num_neighbors=[10, 10], 
                batch_size=batch_size,
                input_nodes=indices
            )
        except Exception as e:
            print(f"NeighborLoader初始化失败: {e}")
            print("使用SimpleBatchLoader替代")
    
    return SimpleBatchLoader(data, batch_size, indices)

def set_seed(seed):
    """设置随机种子以确保结果可重现"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def evaluate_model(model, data_loader, device, num_classes=3):
    """评估模型性能并返回详细指标"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_losses = []
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in data_loader:
            try:
                # 移动数据到设备
                if hasattr(batch, 'to'):
                    batch = batch.to(device)
                else:
                    # 如果使用SimpleBatchLoader
                    batch.x = batch.x.to(device)
                    batch.y = batch.y.to(device)
                    batch.edge_index = batch.edge_index.to(device)
                
                # 获取有效的样本(标签不为-1)
                valid_mask = batch.y != -1
                if valid_mask.sum() == 0:
                    continue
                
                # 前向传播
                outputs = model(batch)
                
                # 获取预测结果
                preds = outputs.argmax(dim=1)
                
                # 计算损失
                loss = criterion(outputs[valid_mask], batch.y[valid_mask])
                
                # 收集结果
                all_labels.extend(batch.y[valid_mask].cpu().numpy())
                all_preds.extend(preds[valid_mask].cpu().numpy())
                all_probs.extend(outputs[valid_mask].cpu().numpy())
                all_losses.extend(loss.cpu().numpy())
                
            except Exception as e:
                print(f"处理批次时出错: {e}")
                continue
    
    # 转换为numpy数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_losses = np.array(all_losses)
    
    # 计算指标
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'avg_loss': np.mean(all_losses),
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    # 计算每个类别的指标
    for i in range(num_classes):
        class_mask = all_labels == i
        if np.any(class_mask):
            metrics[f'precision_class_{i}'] = precision_score(all_labels == i, all_preds == i)
            metrics[f'recall_class_{i}'] = recall_score(all_labels == i, all_preds == i)
            metrics[f'f1_class_{i}'] = f1_score(all_labels == i, all_preds == i)
    
    return metrics, all_labels, all_preds, all_probs, all_losses

def visualize_confusion_matrix(cm, class_names, output_path):
    """可视化混淆矩阵"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_roc_curve(labels, probs, num_classes, output_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    
    # 创建二值化标签用于ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # 将多分类问题转化为二分类问题(one-vs-rest)
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        fpr[i], tpr[i], _ = roc_curve(binary_labels, probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'类别 {i} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('接收者操作特征曲线 (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def visualize_pr_curve(labels, probs, num_classes, output_path):
    """绘制精确率-召回率曲线"""
    plt.figure(figsize=(10, 8))
    
    # 计算每个类别的PR曲线
    for i in range(num_classes):
        binary_labels = (labels == i).astype(int)
        precision, recall, _ = precision_recall_curve(binary_labels, probs[:, i])
        plt.plot(recall, precision, label=f'类别 {i}')
    
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc='best')
    plt.savefig(output_path)
    plt.close()

def visualize_loss_distribution(losses, output_path):
    """可视化损失分布"""
    plt.figure(figsize=(10, 6))
    sns.histplot(losses, bins=50, kde=True)
    plt.xlabel('损失值')
    plt.ylabel('样本数量')
    plt.title('损失分布')
    plt.axvline(x=np.mean(losses), color='r', linestyle='--', label=f'平均损失: {np.mean(losses):.4f}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def visualize_embeddings_2d(model, data_loader, device, output_path, n_samples=1000):
    """使用t-SNE可视化高维嵌入"""
    try:
        from sklearn.manifold import TSNE
        
        # 收集嵌入和标签
        embeddings = []
        labels = []
        
        model.eval()
        with torch.no_grad():
            for batch in data_loader:
                # 移动数据到设备
                if hasattr(batch, 'to'):
                    batch = batch.to(device)
                else:
                    batch.x = batch.x.to(device)
                    batch.y = batch.y.to(device)
                    batch.edge_index = batch.edge_index.to(device)
                
                # 获取嵌入(使用model.get_embedding方法，如果有的话)
                if hasattr(model, 'get_embedding'):
                    emb = model.get_embedding(batch)
                    valid_mask = batch.y != -1
                    embeddings.extend(emb[valid_mask].cpu().numpy())
                    labels.extend(batch.y[valid_mask].cpu().numpy())
                
                # 限制样本数量
                if len(embeddings) >= n_samples:
                    break
        
        if len(embeddings) > 0:
            # 转换为numpy数组
            embeddings = np.array(embeddings)
            labels = np.array(labels)
            
            # 随机采样(如果样本过多)
            if len(embeddings) > n_samples:
                idx = np.random.choice(len(embeddings), n_samples, replace=False)
                embeddings = embeddings[idx]
                labels = labels[idx]
            
            # 使用t-SNE降维
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)
            
            # 可视化
            plt.figure(figsize=(10, 8))
            for i in np.unique(labels):
                mask = labels == i
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'类别 {i}', alpha=0.6)
            
            plt.title('嵌入空间的t-SNE可视化')
            plt.legend()
            plt.savefig(output_path)
            plt.close()
            
            print(f"嵌入可视化已保存到 {output_path}")
        else:
            print("无法获取嵌入用于可视化")
            
    except Exception as e:
        print(f"嵌入可视化失败: {e}")

def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置设备
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 加载数据
    print("加载数据...")
    pretrain_data, finetune_data = load_data_llm(args)
    data = finetune_data  # 使用微调数据进行评估
    
    # 确保测试索引
    if not hasattr(data, 'finetune_test_idx') or data.finetune_test_idx is None:
        # 可以手动设置测试集为15%
        test_ratio = float(args.split_ratio.split(',')[2])
        num_nodes = data.num_nodes
        num_test = int(num_nodes * test_ratio)
        data.finetune_test_idx = torch.arange(num_nodes - num_test, num_nodes)
        print(f"手动设置测试集大小: {len(data.finetune_test_idx)}个节点")
    
    # 创建测试数据加载器
    print("创建测试数据加载器...")
    test_loader = create_data_loader(
        data, 
        args.batch_size, 
        data.finetune_test_idx, 
        use_neighbor_loader=not args.simplified
    )
    
    # 加载模型
    print(f"从 {args.model_path} 加载模型...")
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    try:
        # 加载模型检查点
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 检测模型参数
        if 'llm_layer.weight' in checkpoint:
            args.llm_enhancement_channel = checkpoint['llm_layer.weight'].shape[1]
            print(f"从检查点检测到LLM特征通道维度: {args.llm_enhancement_channel}")
            
        if 'hidden_layer.weight' in checkpoint:
            args.dim = checkpoint['hidden_layer.weight'].shape[0]
            print(f"从检查点检测到嵌入维度: {args.dim}")
        
        # 计算输入维度
        in_dim = None
        if hasattr(data, 'x') and data.x is not None:
            in_dim = data.x.shape[1]
            print(f"从数据检测到输入维度: {in_dim}")
        else:
            # 从checkpoint估计
            if 'num_layer.weight' in checkpoint:
                in_dim = checkpoint['num_layer.weight'].shape[1]
                print(f"从检查点估计输入维度: {in_dim}")
            else:
                # 默认值
                in_dim = 128
                print(f"使用默认输入维度: {in_dim}")
        
        # 创建模型
        model = SeGA_LLM(
            in_dim=in_dim,
            embed_dim=args.dim,
            dropout=0.3,
            temp=args.temp,
            pretext_task=args.pretext_task,
            pretrain=False,
            device=device,
            encoder=args.encoder,
            args=args
        )
        
        # 加载模型权重
        model.load_state_dict(checkpoint)
        print("模型加载成功!")
        
        # 将模型移至设备
        model = model.to(device)
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 评估模型
    print("开始评估模型...")
    metrics, labels, preds, probs, losses = evaluate_model(model, test_loader, device)
    
    # 打印指标
    print("\n===== 评估结果 =====")
    print(f"准确率: {metrics['accuracy']:.4f}")
    print(f"精确率: {metrics['precision']:.4f}")
    print(f"召回率: {metrics['recall']:.4f}")
    print(f"F1分数: {metrics['f1']:.4f}")
    print(f"平衡准确率: {metrics['balanced_accuracy']:.4f}")
    print(f"平均损失: {metrics['avg_loss']:.4f}")
    
    # 保存指标到文本文件
    result_path = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(result_path, 'w') as f:
        f.write("===== 模型评估结果 =====\n")
        f.write(f"准确率: {metrics['accuracy']:.4f}\n")
        f.write(f"精确率: {metrics['precision']:.4f}\n")
        f.write(f"召回率: {metrics['recall']:.4f}\n")
        f.write(f"F1分数: {metrics['f1']:.4f}\n")
        f.write(f"平衡准确率: {metrics['balanced_accuracy']:.4f}\n")
        f.write(f"平均损失: {metrics['avg_loss']:.4f}\n\n")
        
        f.write("类别详细指标:\n")
        for i in range(3):  # 假设有3个类别
            if f'precision_class_{i}' in metrics:
                f.write(f"类别 {i}:\n")
                f.write(f"  精确率: {metrics[f'precision_class_{i}']:.4f}\n")
                f.write(f"  召回率: {metrics[f'recall_class_{i}']:.4f}\n")
                f.write(f"  F1分数: {metrics[f'f1_class_{i}']:.4f}\n")
        
        f.write("\n混淆矩阵:\n")
        f.write(str(metrics['confusion_matrix']))
    
    print(f"评估结果已保存至: {result_path}")
    
    # 可视化结果
    print("\n生成可视化图表...")
    
    # 定义类别名称
    class_names = ['人类', '机器人', '水军']
    
    # 混淆矩阵
    visualize_confusion_matrix(
        metrics['confusion_matrix'], 
        class_names, 
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # ROC曲线
    visualize_roc_curve(
        labels, 
        probs, 
        len(class_names), 
        os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # 精确率-召回率曲线
    visualize_pr_curve(
        labels, 
        probs, 
        len(class_names), 
        os.path.join(args.output_dir, 'pr_curve.png')
    )
    
    # 损失分布
    visualize_loss_distribution(
        losses, 
        os.path.join(args.output_dir, 'loss_distribution.png')
    )
    
    # 嵌入可视化
    visualize_embeddings_2d(
        model, 
        test_loader, 
        device, 
        os.path.join(args.output_dir, 'embeddings_visualization.png')
    )
    
    print(f"所有可视化结果已保存到 {args.output_dir}")
    print("评估完成!")

if __name__ == "__main__":
    main() 