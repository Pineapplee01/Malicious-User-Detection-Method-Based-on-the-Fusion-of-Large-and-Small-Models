import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import pytorch_lightning as pl
from torch_geometric.loader import ClusterData, ClusterLoader

# 导入自定义模块
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="模型ROC和PR曲线评估")
    
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

def plot_roc_curve(y_true, y_score, n_classes, output_path):
    """绘制ROC曲线"""
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算每个类别的ROC曲线和AUC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算微平均ROC曲线和AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 绘制所有ROC曲线
    plt.figure(figsize=(10, 8))
    
    # 绘制对角线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # 绘制每个类别的ROC曲线
    colors = ['blue', 'red', 'green']
    class_names = ['人类', '机器人', '水军']
    
    for i, color, name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc[i]:.4f})')
    
    # 绘制微平均ROC曲线
    plt.plot(fpr["micro"], tpr["micro"], 'darkorange', lw=2,
             label=f'微平均 (AUC = {roc_auc["micro"]:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('接收者操作特征(ROC)曲线')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return roc_auc

def plot_pr_curve(y_true, y_score, n_classes, output_path):
    """绘制精确率-召回率曲线"""
    # 二值化标签
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    # 计算每个类别的PR曲线
    precision = dict()
    recall = dict()
    average_precision = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_true_bin[:, i], y_score[:, i])
    
    # 计算微平均PR曲线
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_true_bin.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(y_true_bin.ravel(), y_score.ravel())
    
    # 绘制PR曲线
    plt.figure(figsize=(10, 8))
    
    # 绘制每个类别的PR曲线
    colors = ['blue', 'red', 'green']
    class_names = ['人类', '机器人', '水军']
    
    for i, color, name in zip(range(n_classes), colors, class_names):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'{name} (AP = {average_precision[i]:.4f})')
    
    # 绘制微平均PR曲线
    plt.plot(recall["micro"], precision["micro"], 'darkorange', lw=2,
             label=f'微平均 (AP = {average_precision["micro"]:.4f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('召回率')
    plt.ylabel('精确率')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return average_precision

def get_predictions(model, data, args):
    """获取模型预测和真实标签"""
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
    all_probs = []
    all_labels = []
    
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
            logits = result["logits"]
            
            # 计算概率
            probs = torch.softmax(logits, dim=1)
            
            # 仅考虑测试集样本
            batch_labels = batch.y[batch_test_mask]
            batch_probs = probs[batch_test_mask]
            
            all_probs.extend(batch_probs.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    # 转换为numpy数组
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    return all_labels, all_probs

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
    
    # 获取模型预测
    print("正在获取模型预测...")
    y_true, y_score = get_predictions(model, finetune_data, args)
    
    # 类别数
    n_classes = 3  # 假设有3个类别：人类、机器人和水军
    
    # 绘制ROC曲线
    print("正在绘制ROC曲线...")
    roc_path = os.path.join(args.output_dir, 'roc_curve.png')
    roc_auc = plot_roc_curve(y_true, y_score, n_classes, roc_path)
    print(f"ROC曲线已保存至: {roc_path}")
    
    # 绘制PR曲线
    print("正在绘制PR曲线...")
    pr_path = os.path.join(args.output_dir, 'pr_curve.png')
    avg_precision = plot_pr_curve(y_true, y_score, n_classes, pr_path)
    print(f"PR曲线已保存至: {pr_path}")
    
    # 保存AUC和AP值到文本文件
    curves_txt_path = os.path.join(args.output_dir, 'curves_metrics.txt')
    with open(curves_txt_path, 'w') as f:
        f.write("----- ROC曲线和PR曲线指标 -----\n")
        f.write("各类别ROC曲线下面积(AUC):\n")
        for i, name in enumerate(['人类', '机器人', '水军']):
            f.write(f"{name}: {roc_auc[i]:.4f}\n")
        f.write(f"微平均: {roc_auc['micro']:.4f}\n\n")
        
        f.write("各类别平均精度(AP):\n")
        for i, name in enumerate(['人类', '机器人', '水军']):
            f.write(f"{name}: {avg_precision[i]:.4f}\n")
        f.write(f"微平均: {avg_precision['micro']:.4f}\n")
    
    print(f"曲线指标已保存至: {curves_txt_path}")
    print("\n评估完成!")

if __name__ == "__main__":
    main() 