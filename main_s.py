import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import os
import sys
import numpy as np
import random
import json
import time
import math
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm
import globals
from datetime import datetime
from tqdm import tqdm

# ===== 简易配置区域 =====
# 在这里修改配置，无需命令行参数
CONFIG = {
    # 数据路径
    "dataset_path": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "processed_data")),
    # 模型路径（用于加载或保存）
    "model_path": "./checkpoints/20250410-171222_finetune_best_model.pt",
    # 输出目录
    "output_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evaluation_results")),
    # 模型参数
    "dim": 16,                      # 嵌入维度
    "encoder": "SimpleGCN",         # 编码器类型
    "prompt_encoder": "roberta",    # 提示编码器类型
    "llm_model": "mistral",         # LLM模型类型
    # 训练参数
    "batch_size": 8,                # 批次大小
    "test_batch_size": 8,           # 测试批次大小
    "epoch": 10,                     # 训练轮次
    "lr": 0.001,                    # 学习率
    "dropout": 0.3,                 # Dropout比例
    # 其他设置
    "cuda": 0,                      # GPU ID
    "seed": 42,                     # 随机种子
    "simplified": True,             # 使用简化模型（不依赖PyG）
    "debug": True,                  # 调试模式
    "generate_plots": True,         # 生成图表
    # 额外的LLM配置
    "llm_enhancement_channel": 100, # LLM增强通道维度
    "lst": True,                    # 是否使用列表特征
    "pretext_task": "contrastive",  # 预训练任务类型
    "template": "l",                # 模板选择
    "edge_types": "all"             # 边类型
}
# ===== 配置区域结束 =====

# 确保目录添加到路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', 'SeGA-main', 'Code'))

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SeGA-main/Code')))

# 导入自定义模块
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm

# 定义用于保存和绘制指标的类
class MetricsTracker:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建图表输出目录
        self.plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # 初始化指标记录
        self.metrics = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'train_f1': [],
            'val_f1': []
        }
        
        # 初始化测试结果
        self.test_results = None
        
        # 初始化预测结果
        self.predictions = {
            'true': [],
            'pred': [],
            'probs': []
        }
    
    def update_metrics(self, epoch, train_metrics, val_metrics):
        """更新每个epoch的训练和验证指标"""
        self.metrics['epoch'].append(epoch)
        
        for phase in ['train', 'val']:
            metrics = train_metrics if phase == 'train' else val_metrics
            
            for metric in ['loss', 'acc', 'f1']:
                if metric in metrics:
                    self.metrics[f'{phase}_{metric}'].append(metrics[metric])
                else:
                    # 如果指标不存在，添加NaN
                    self.metrics[f'{phase}_{metric}'].append(float('nan'))
        
        # 保存当前指标
        self.save_metrics()
    
    def update_test_results(self, results):
        """更新测试结果"""
        self.test_results = results
        
        # 保存测试结果
        results_path = os.path.join(self.output_dir, 'test_results.txt')
        with open(results_path, 'w') as f:
            f.write("===== 模型测试结果 =====\n")
            for metric, value in results.items():
                if metric != 'confusion_matrix':
                    f.write(f"{metric}: {value:.4f}\n")
            
            if 'confusion_matrix' in results:
                f.write("\n混淆矩阵:\n")
                f.write(str(results['confusion_matrix']))
        
        # 保存混淆矩阵
        if 'confusion_matrix' in results:
            np.save(os.path.join(self.output_dir, 'confusion_matrix.npy'), results['confusion_matrix'])
    
    def update_predictions(self, y_true, y_pred, y_probs=None):
        """更新预测结果"""
        self.predictions['true'].extend(y_true)
        self.predictions['pred'].extend(y_pred)
        
        if y_probs is not None:
            if not self.predictions['probs']:
                self.predictions['probs'] = y_probs
            else:
                self.predictions['probs'] = np.vstack([self.predictions['probs'], y_probs])
        
        # 保存预测结果
        np.savez(
            os.path.join(self.output_dir, 'predictions.npz'),
            labels=np.array(self.predictions['true']),
            preds=np.array(self.predictions['pred']),
            probs=np.array(self.predictions['probs']) if len(self.predictions['probs']) > 0 else None
        )
    
    def save_metrics(self):
        """保存训练指标到CSV文件"""
        df = pd.DataFrame(self.metrics)
        df.to_csv(os.path.join(self.output_dir, 'training_metrics.csv'), index=False)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if len(self.metrics['epoch']) == 0:
            return
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics['epoch'], self.metrics['train_loss'], 'b-', label='训练损失')
        plt.plot(self.metrics['epoch'], self.metrics['val_loss'], 'r-', label='验证损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics['epoch'], self.metrics['train_acc'], 'b-', label='训练准确率')
        plt.plot(self.metrics['epoch'], self.metrics['val_acc'], 'r-', label='验证准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('训练和验证准确率')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'))
        plt.close()
    
    def plot_confusion_matrix(self, class_names=None):
        """绘制混淆矩阵"""
        if self.test_results is None or 'confusion_matrix' not in self.test_results:
            return
        
        cm = self.test_results['confusion_matrix']
        if class_names is None:
            class_names = ['人类', '机器人', '水军'][:len(cm)]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrix.png'))
        plt.close()
    
    def plot_roc_curves(self, num_classes=3):
        """绘制ROC曲线"""
        if len(self.predictions['true']) == 0 or len(self.predictions['probs']) == 0:
            return
        
        y_true = np.array(self.predictions['true'])
        y_score = np.array(self.predictions['probs'])
        
        plt.figure(figsize=(10, 8))
        
        for i in range(num_classes):
            # 使用one-vs-rest策略
            binary_labels = (y_true == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, y_score[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'类别 {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率 (FPR)')
        plt.ylabel('真正例率 (TPR)')
        plt.title('接收者操作特征曲线 (ROC)')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(self.plots_dir, 'roc_curves.png'))
        plt.close()
    
    def plot_pr_curves(self, num_classes=3):
        """绘制精确率-召回率曲线"""
        if len(self.predictions['true']) == 0 or len(self.predictions['probs']) == 0:
            return
        
        y_true = np.array(self.predictions['true'])
        y_score = np.array(self.predictions['probs'])
        
        plt.figure(figsize=(10, 8))
        
        for i in range(num_classes):
            binary_labels = (y_true == i).astype(int)
            precision, recall, _ = precision_recall_curve(binary_labels, y_score[:, i])
            plt.plot(recall, precision, label=f'类别 {i}')
        
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.plots_dir, 'pr_curves.png'))
        plt.close()
    
    def plot_class_distribution(self):
        """绘制类别分布图"""
        if len(self.predictions['true']) == 0:
            return
        
        labels = np.array(self.predictions['true'])
        class_counts = np.bincount(labels)
        class_names = ['人类', '机器人', '水军'][:len(class_counts)]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(class_counts)), class_counts, tick_label=class_names)
        plt.xlabel('类别')
        plt.ylabel('样本数量')
        plt.title('类别分布')
        
        # 在柱状图上显示数量
        for i, count in enumerate(class_counts):
            plt.text(i, count + 5, str(count), ha='center')
        
        plt.savefig(os.path.join(self.plots_dir, 'class_distribution.png'))
        plt.close()
        
        # 绘制饼图
        plt.figure(figsize=(8, 8))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('类别分布(百分比)')
        plt.savefig(os.path.join(self.plots_dir, 'class_distribution_pie.png'))
        plt.close()
    
    def generate_report(self):
        """生成包含所有图表的HTML报告"""
        # 保存所有图表
        self.plot_training_curves()
        self.plot_confusion_matrix()
        self.plot_roc_curves()
        self.plot_pr_curves()
        self.plot_class_distribution()
        
        # 创建HTML报告
        report_path = os.path.join(self.output_dir, 'evaluation_report.html')
        
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>SeGA-LLM 评估报告</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333366; }
                h2 { color: #666699; margin-top: 30px; }
                img { max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }
                .image-container { margin-bottom: 30px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                th { background-color: #4CAF50; color: white; }
            </style>
        </head>
        <body>
            <h1>SeGA-LLM 模型评估报告</h1>
        """
        
        # 添加测试指标
        if self.test_results is not None:
            html += """
            <h2>性能指标</h2>
            <table>
                <tr><th>指标</th><th>值</th></tr>
            """
            
            for metric, value in self.test_results.items():
                if metric != 'confusion_matrix':
                    html += f"<tr><td>{metric}</td><td>{value:.4f}</td></tr>"
            
            html += "</table>"
        
        # 添加图片
        plots = [
            ('training_curves.png', '训练曲线'),
            ('confusion_matrix.png', '混淆矩阵'),
            ('roc_curves.png', 'ROC曲线'),
            ('pr_curves.png', '精确率-召回率曲线'),
            ('class_distribution.png', '类别分布'),
            ('class_distribution_pie.png', '类别分布(饼图)')
        ]
        
        for plot_file, plot_title in plots:
            plot_path = os.path.join('plots', plot_file)
            if os.path.exists(os.path.join(self.plots_dir, plot_file)):
                html += f"""
                <div class="image-container">
                    <h2>{plot_title}</h2>
                    <img src="{plot_path}" alt="{plot_title}">
                </div>
                """
        
        html += """
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"评估报告已保存至 {report_path}")

def parse_args():
    # 仅为向后兼容保留此函数
    parser = argparse.ArgumentParser(description='SeGA-LLM')
    parser.add_argument("--dataset_path", type=str, default=CONFIG["dataset_path"],
                        help="处理后数据的路径")
    parser.add_argument('--dim', type=int, default=CONFIG["dim"],
                        help='embedding dimensions')
    parser.add_argument('--cuda', type=int, default=CONFIG["cuda"],
                        help='cuda device')
    parser.add_argument('--template', type=str, default=CONFIG["template"],
                        help='template choice')
    parser.add_argument('--list_num', type=int, default=50,
                        help='number of lists')
    parser.add_argument('--epoch', type=int, default=CONFIG["epoch"],
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=CONFIG["batch_size"],
                        help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=CONFIG["test_batch_size"],
                        help='test batch size')
    parser.add_argument('--encoder', type=str, default=CONFIG["encoder"],
                        help='choice of encoder')
    parser.add_argument('--prompt_encoder', type=str, default=CONFIG["prompt_encoder"],
                        help='choice of prompt encoder')
    parser.add_argument('--sample', action='store_true', default=False,
                        help='if sampling neighbors')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='whether to pretrain')
    parser.add_argument('--seed', type=int, default=CONFIG["seed"],
                        help='random seed')
    parser.add_argument('--edge_types', type=str, default=CONFIG["edge_types"],
                        help='types of edges')
    parser.add_argument('--lr', type=float, default=CONFIG["lr"],
                        help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--dropout', type=float, default=CONFIG["dropout"],
                        help='dropout probability')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for contrastive loss')
    parser.add_argument('--pretext_task', type=str, default=CONFIG["pretext_task"],
                        help='pretext task')
    parser.add_argument('--lst', action='store_true', default=CONFIG["lst"],
                        help='if using list features')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='log directory for tensorboard')
    parser.add_argument('--output_dir', type=str, default=CONFIG["output_dir"],
                        help='output directory for saving models')
    parser.add_argument('--llm_model', type=str, default=CONFIG["llm_model"],
                        help='LLM model type to use (mistral/llama/gemma)')
    parser.add_argument('--llm_enhancement_channel', type=int, default=CONFIG["llm_enhancement_channel"],
                        help='LLM enhancement channel dimension')
    parser.add_argument('--checkpoint', type=str, default=CONFIG["model_path"],
                        help='Path to load pretrained model checkpoint')
    parser.add_argument('--simplified', action='store_true', default=CONFIG["simplified"],
                        help='Use simplified model without PyG dependencies')
    parser.add_argument('--debug', action='store_true', default=CONFIG["debug"],
                        help='Run in debug mode with minimal data')
    parser.add_argument('--split_ratio', type=str, default="0.8,0.05,0.15",
                        help='训练:验证:测试的比例(默认80%训练,5%验证,15%测试)')
    parser.add_argument('--generate_plots', action='store_true', default=CONFIG["generate_plots"],
                        help='生成评估图表')
    return parser.parse_args()

def set_seed(seed):
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
                batch = batch.to(device)
                
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
        if i in all_labels:
            metrics[f'precision_class_{i}'] = precision_score(all_labels == i, all_preds == i)
            metrics[f'recall_class_{i}'] = recall_score(all_labels == i, all_preds == i)
            metrics[f'f1_class_{i}'] = f1_score(all_labels == i, all_preds == i)
    
    return metrics, all_labels, all_preds, all_probs, all_losses

def train(args, pretrain, finetune, model_path):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化全局变量存储测试结果
    globals.init()
    
    # 初始化指标跟踪器
    metrics_tracker = MetricsTracker(args.output_dir)
    
    # 准备设备
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    
    # 预训练阶段
    if args.pretrain:
        print("开始预训练阶段...")
        # 加载数据
        train_loader = NeighborLoader(pretrain, num_neighbors=[10, 10], batch_size=args.batch_size,
                                     input_nodes=pretrain.pretrain_train_idx)
        valid_loader = NeighborLoader(pretrain, num_neighbors=[10, 10], batch_size=args.batch_size,
                                     input_nodes=pretrain.pretrain_valid_idx)
        test_loader = NeighborLoader(pretrain, num_neighbors=[10, 10], batch_size=args.test_batch_size,
                                    input_nodes=pretrain.pretrain_test_idx)
        
        model = SeGA_LLM(args, pretrain=True)
        
        # 设置保存最佳模型的回调
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='pretrain-{epoch:02d}-{val_acc:.4f}',
            monitor='val_acc',
            mode='max',
            save_top_k=1
        )
        
        # 设置训练器
        trainer = pl.Trainer(
            max_epochs=args.pretrain_epochs,
            accelerator="gpu",
            devices=args.devices,
            callbacks=[checkpoint_callback],
            default_root_dir=args.output_dir
        )
        
        # 训练模型
        trainer.fit(model, train_loader, valid_loader)
        
        # 评估模型
        trainer.test(model, test_loader)
        
        # 保存预训练模型路径
        model_path = checkpoint_callback.best_model_path
        print(f"预训练完成，最佳模型保存于: {model_path}")
    
    # 微调阶段
    print("开始微调阶段...")
    # 加载数据
    train_loader = NeighborLoader(finetune, num_neighbors=[10, 10], batch_size=args.batch_size,
                                 input_nodes=finetune.finetune_train_idx)
    valid_loader = NeighborLoader(finetune, num_neighbors=[10, 10], batch_size=args.batch_size,
                                 input_nodes=finetune.finetune_valid_idx)
    test_loader = NeighborLoader(finetune, num_neighbors=[10, 10], batch_size=args.test_batch_size,
                                input_nodes=finetune.finetune_test_idx)
    
    # 初始化模型
    if args.pretrain and model_path:
        # 从预训练模型加载
        print(f"从预训练模型加载: {model_path}")
        model = SeGA_LLM.load_from_checkpoint(model_path, args=args, pretrain=False)
    else:
        # 从头开始训练
        model = SeGA_LLM(
            in_dim=finetune.x.shape[1],
            embed_dim=args.dim,
            dropout=args.dropout,
            temp=args.temp,
            pretext_task=args.pretext_task,
            pretrain=False,
            device=device,
            encoder=args.encoder,
            args=args
        )
    
    # 将模型移至设备
    model = model.to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    best_val_f1 = 0
    best_model_state = None
    
    print(f"开始训练，共{args.epoch}个轮次...")
    for epoch in range(args.epoch):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} (Train)"):
            # 移动数据到设备
            batch = batch.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(batch)
            
            # 计算损失(只考虑有标签的节点)
            valid_mask = batch.y != -1
            if valid_mask.sum() == 0:
                continue
                
            loss = criterion(outputs[valid_mask], batch.y[valid_mask])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 记录损失和预测
            train_loss += loss.item()
            preds = outputs[valid_mask].argmax(dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(batch.y[valid_mask].cpu().numpy())
        
        # 计算训练指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_metrics, val_labels, val_preds, val_probs, val_losses = evaluate_model(model, valid_loader, device)
        
        # 更新指标跟踪器
        metrics_tracker.update_metrics(
            epoch,
            {'loss': train_loss, 'acc': train_acc, 'f1': train_f1},
            val_metrics
        )
        
        # 打印当前轮次结果
        print(f"Epoch {epoch+1}/{args.epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Valid Loss: {val_metrics['avg_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            print(f"  找到新的最佳模型! F1: {best_val_f1:.4f}")
            
            # 保存最佳模型
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            best_model_path = os.path.join(args.output_dir, f"{timestamp}_finetune_best_model.pt")
            torch.save(best_model_state, best_model_path)
            print(f"  最佳模型已保存到: {best_model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 加载最佳模型进行测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试阶段
    print("\n开始在测试集上评估最佳模型...")
    test_metrics, test_labels, test_preds, test_probs, test_losses = evaluate_model(model, test_loader, device)
    
    # 更新指标跟踪器
    metrics_tracker.update_test_results(test_metrics)
    metrics_tracker.update_predictions(test_labels, test_preds, test_probs)
    
    # 打印测试结果
    print("\n===== 测试结果 =====")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"召回率: {test_metrics['recall']:.4f}")
    print(f"平衡准确率: {test_metrics['balanced_accuracy']:.4f}")
    
    # 生成各种图表
    if args.generate_plots:
        print("\n生成评估图表...")
        metrics_tracker.generate_report()
    
    # 更新全局变量
    globals.acc = test_metrics['accuracy']
    globals.f1 = test_metrics['f1']
    globals.precision = test_metrics['precision']
    globals.recall = test_metrics['recall']
    globals.bacc = test_metrics['balanced_accuracy']
    
    return test_metrics

# 简化的数据加载函数
def load_data(args):
    """
    加载原始数据，不区分预训练和微调
    """
    print("使用统一数据加载...")
    pretrain_data, finetune_data = load_data_llm(args)
    return finetune_data  # 只返回微调数据

def load_lst_data(args):
    """
    加载包含列表特征的数据
    """
    print("加载包含列表特征的统一数据...")
    args.lst = True  # 确保使用列表特征
    pretrain_data, finetune_data = load_data_llm(args)
    return finetune_data  # 只返回微调数据

def process_edge_index(data, verbose=True):
    """
    处理边索引，确保其格式正确并且不包含超出节点范围的索引
    
    参数:
        data: 包含edge_index的数据对象
        verbose: 是否打印详细信息
    
    返回:
        处理后的数据对象
    """
    if not hasattr(data, 'edge_index') or data.edge_index is None:
        if verbose:
            print("警告: 数据中没有edge_index属性")
        return data
        
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # 记录原始形状
    original_shape = edge_index.shape
    if verbose:
        print(f"原始边索引形状: {original_shape}")
    
    # 检查并转置边索引
    if edge_index.shape[0] != 2:
        if verbose:
            print(f"调整边索引形状: {edge_index.shape} -> [2, num_edges]")
        edge_index = edge_index.t()
        data.edge_index = edge_index
    
    # 检查边索引是否超出有效范围
    row, col = edge_index[0], edge_index[1]
    out_of_bounds_mask = (row >= num_nodes) | (col >= num_nodes)
    
    if out_of_bounds_mask.any():
        invalid_edges = out_of_bounds_mask.sum().item()
        if verbose:
            print(f"发现{invalid_edges}条边索引超出节点范围，正在剪裁...")
        
        # 剪裁超出范围的边索引
        data.edge_index[0] = torch.clamp(edge_index[0], 0, num_nodes - 1)
        data.edge_index[1] = torch.clamp(edge_index[1], 0, num_nodes - 1)
        
        if verbose:
            print("边索引剪裁完成")
            
    # 打印最终边索引信息
    if verbose:
        print(f"处理后边索引形状: {data.edge_index.shape}, 节点数量: {num_nodes}")
        
    return data

def main():
    time_start = time.time()
    
    # 从配置创建参数对象
    args = type('Args', (), {})()
    
    # 将CONFIG中的所有键值对复制到args对象
    for key, value in CONFIG.items():
        setattr(args, key, value)
    
    # 设置一些可能缺失的默认值
    if not hasattr(args, 'wd'):
        setattr(args, 'wd', 0.0001)
    if not hasattr(args, 'temp'):
        setattr(args, 'temp', 0.05)
    if not hasattr(args, 'log_dir'):
        setattr(args, 'log_dir', './runs')
    if not hasattr(args, 'num_workers'):
        setattr(args, 'num_workers', 0)
    if not hasattr(args, 'pretrain'):
        setattr(args, 'pretrain', False)
    if not hasattr(args, 'sample'):
        setattr(args, 'sample', False)
    if not hasattr(args, 'list_num'):
        setattr(args, 'list_num', 50)
    
    print("\n===== SeGA-LLM 开始运行 =====")
    print(f"模型路径: {args.model_path}")
    print(f"数据路径: {args.dataset_path}")
    print(f"输出目录: {args.output_dir}")
    print(f"嵌入维度: {args.dim}")
    print(f"编码器: {args.encoder}")
    print(f"简化模式: {args.simplified}")
    print(f"预训练任务: {args.pretext_task}")
    print(f"模板: {args.template}")
    print("============================\n")
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    if torch.cuda.is_available():
        device = f"cuda:{args.cuda}"
    else:
        device = "cpu"
    print(f"使用设备: {device}")
    
    # 简化模式通知
    if args.simplified:
        print("使用简化模式，不依赖PyG库")
        args.encoder = 'SimpleGCN'
    
    # 调试模式
    if args.debug:
        print("DEBUG模式: 使用最小批量和少量数据")
        args.batch_size = 2
        args.epoch = 1
        args.test_batch_size = 2
        # 减小嵌入维度以降低内存使用
        args.dim = 16
        print(f"调整参数: batch_size={args.batch_size}, epoch={args.epoch}, dim={args.dim}")
        
    # 创建输出目录 - 使用规范化路径
    output_dir = os.path.normpath(os.path.abspath(args.output_dir))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 创建TensorBoard日志目录 - 修复Windows路径问题
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.normpath(os.path.join(os.path.abspath(getattr(args, 'log_dir', './runs')), run_id))
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"已创建TensorBoard日志目录: {log_dir}")
    except Exception as e:
        print(f"创建日志目录失败: {e}，尝试使用备用目录")
        # 使用备用日志目录
        log_dir = os.path.normpath(os.path.join(output_dir, "tb_logs", run_id))
        os.makedirs(log_dir, exist_ok=True)
        print(f"使用备用日志目录: {log_dir}")
    
    # 创建TensorBoardWriter
    try:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard日志将保存到 {log_dir}")
    except ImportError as e:
        print(f"无法导入TensorBoard: {e}，将使用虚拟Writer")
        # 创建一个虚拟的writer
        class DummyWriter:
            def add_scalar(self, *args, **kwargs): pass
            def close(self): pass
        writer = DummyWriter()
    except Exception as e:
        print(f"TensorBoard初始化失败: {e}，将使用虚拟Writer")
        # 出错时使用虚拟Writer
        class DummyWriter:
            def add_scalar(self, *args, **kwargs): pass
            def close(self): pass
        writer = DummyWriter()
    
    # 创建指标跟踪器
    metrics_tracker = MetricsTracker(output_dir)
    
    # 加载数据 - 修正数据加载逻辑
    print(f"加载LLM({args.llm_model})增强的数据...")
    try:
        # 直接使用load_data_llm，获取原始数据
        pretrain_data, finetune_data = load_data_llm(args)
        print(f"数据加载成功")
    except Exception as e:
        print(f"数据加载出错: {e}")
        # 出错时重新尝试
        print("尝试重新加载...")
        from data_loader_llm import load_data_llm as load_data_retry
        pretrain_data, finetune_data = load_data_retry(args)
    
    # 选择要使用的数据
    data = finetune_data if not args.pretrain else pretrain_data
    
    # 打印原始数据信息
    print(f"原始数据: 节点数量={data.num_nodes}, 特征维度={data.x.shape}")
    if hasattr(data, 'edge_index') and data.edge_index is not None:
        print(f"边数量={data.edge_index.shape[1]}")
    
    # 预处理边索引
    print("预处理图边索引...")
    data = process_edge_index(data, verbose=True)
    
    # 重新分割数据集（如果没有预设的分割）
    if not hasattr(data, 'finetune_train_idx') or not hasattr(data, 'finetune_valid_idx') or not hasattr(data, 'finetune_test_idx'):
        print(f"重新分割数据集，使用比例 {args.split_ratio}...")
        
        # 解析分割比例
        train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split(','))
        print(f"训练集: {train_ratio*100:.1f}%, 验证集: {val_ratio*100:.1f}%, 测试集: {test_ratio*100:.1f}%")
        
        # 获取有标签的节点索引
        labeled_mask = data.y != -1
        labeled_indices = torch.nonzero(labeled_mask, as_tuple=True)[0]
        
        # 打乱索引
        perm = torch.randperm(len(labeled_indices))
        labeled_indices = labeled_indices[perm]
        
        # 计算分割点
        train_end = int(len(labeled_indices) * train_ratio)
        val_end = train_end + int(len(labeled_indices) * val_ratio)
        
        # 分割数据
        data.finetune_train_idx = labeled_indices[:train_end]
        data.finetune_valid_idx = labeled_indices[train_end:val_end]
        data.finetune_test_idx = labeled_indices[val_end:]
        
        # 打印分割情况
        print(f"训练集: {len(data.finetune_train_idx)}个样本")
        print(f"验证集: {len(data.finetune_valid_idx)}个样本")
        print(f"测试集: {len(data.finetune_test_idx)}个样本")
    
    # 清理不必要的数据节省内存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 创建数据加载器
    print("创建数据加载器...")
    try:
        train_loader = NeighborLoader(
            data,
            num_neighbors=[10, 10],
            batch_size=args.batch_size,
            input_nodes=data.finetune_train_idx,
            shuffle=True
        )
        
        valid_loader = NeighborLoader(
            data,
            num_neighbors=[10, 10],
            batch_size=args.batch_size,
            input_nodes=data.finetune_valid_idx
        )
        
        test_loader = NeighborLoader(
            data,
            num_neighbors=[10, 10],
            batch_size=args.test_batch_size,
            input_nodes=data.finetune_test_idx
        )
        
        print("成功创建数据加载器")
    except Exception as e:
        print(f"创建NeighborLoader失败: {e}")
        print("使用简化的批处理替代")
        
        # 自定义简单批处理
        class SimpleBatchLoader:
            """简单的批处理加载器，不依赖PyG的NeighborLoader"""
            def __init__(self, data, batch_size, input_nodes, shuffle=False):
                self.data = data
                self.batch_size = batch_size
                self.input_nodes = input_nodes
                self.shuffle = shuffle
                self.current = 0
                
            def __iter__(self):
                self.current = 0
                if self.shuffle:
                    indices = self.input_nodes.clone()
                    perm = torch.randperm(len(indices))
                    self.shuffled_indices = indices[perm]
                else:
                    self.shuffled_indices = self.input_nodes
                return self
                
            def __next__(self):
                if self.current >= len(self.shuffled_indices):
                    raise StopIteration
                    
                end = min(self.current + self.batch_size, len(self.shuffled_indices))
                batch_indices = self.shuffled_indices[self.current:end]
                self.current = end
                
                batch = type('BatchObject', (), {})()
                batch.y = self.data.y[batch_indices]
                batch.to = lambda device: batch  # 模拟to方法
                batch.x = self.data.x[batch_indices]
                
                # 添加列表特征(如果存在)
                if hasattr(self.data, 'x_list') and self.data.x_list is not None:
                    batch.x_list = self.data.x_list
                    
                # 创建批次索引    
                batch.batch = torch.zeros(len(batch_indices), dtype=torch.long)
                batch.ptr = torch.tensor([0, len(batch_indices)], dtype=torch.long)
                
                # 处理边索引 - 更健壮的方式
                if hasattr(self.data, 'edge_index') and self.data.edge_index is not None:
                    try:
                        # 筛选出与当前批次中节点相关的边
                        edge_index = self.data.edge_index
                        
                        # 根据数据节点总数创建子图边
                        num_nodes = self.data.num_nodes
                        batch.edge_index = torch.zeros((2, 0), dtype=torch.long)
                        
                        # 安全的边索引处理
                        if edge_index.shape[0] == 2 and edge_index.shape[1] > 0:
                            # 确保边索引在有效范围内
                            valid_edges_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes)
                            if valid_edges_mask.sum() > 0:
                                batch.edge_index = edge_index[:, valid_edges_mask]
                        
                        # 裁剪边索引到有效范围
                        batch.edge_index = torch.clamp(batch.edge_index, 0, len(batch_indices) - 1)
                        
                    except Exception as e:
                        print(f"处理边索引时出错: {e}")
                        # 出错时使用空边
                        batch.edge_index = torch.zeros((2, 0), dtype=torch.long)
                else:
                    # 如果数据中没有边索引，添加空的
                    batch.edge_index = torch.zeros((2, 0), dtype=torch.long)
                
                return batch
            
            def __len__(self):
                return (len(self.input_nodes) + self.batch_size - 1) // self.batch_size
        
        # 创建简化的数据加载器
        train_loader = SimpleBatchLoader(data, args.batch_size, data.finetune_train_idx, shuffle=True)
        valid_loader = SimpleBatchLoader(data, args.batch_size, data.finetune_valid_idx)
        test_loader = SimpleBatchLoader(data, args.test_batch_size, data.finetune_test_idx)
        
        print("已创建简化批处理加载器")
    
    # 创建模型
    print(f"创建模型: 输入维度={data.x.shape[1]}, 嵌入维度={args.dim}")
    model = SeGA_LLM(
        in_dim=data.x.shape[1],
        embed_dim=args.dim,
        dropout=args.dropout,
        temp=args.temp,
        pretext_task=args.pretext_task,
        pretrain=args.pretrain,
        device=device,
        encoder=args.encoder,
        args=args
    )
    
    # 如果指定了检查点，加载预训练模型
    if args.checkpoint is not None:
        try:
            checkpoint_path = os.path.abspath(args.checkpoint)
            print(f"加载检查点 {checkpoint_path}")
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        except Exception as e:
            print(f"加载检查点失败: {e}")
    
    # 移动模型到设备
    model = model.to(device)
    
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 使用内存节省技术
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 训练循环
    print(f"开始训练，共{args.epoch}个轮次...")
    best_val_f1 = 0
    best_model_state = None
    
    for epoch in range(args.epoch):
        # 训练阶段
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epoch} (Train)"):
            try:
                # 移动数据到设备
                batch.x = batch.x.to(device)
                batch.y = batch.y.to(device)
                if hasattr(batch, 'edge_index') and batch.edge_index is not None:
                    batch.edge_index = batch.edge_index.to(device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(batch)
                
                # 计算损失(只考虑有标签的节点)
                valid_mask = batch.y != -1
                if valid_mask.sum() == 0:
                    continue
                    
                loss = criterion(outputs[valid_mask], batch.y[valid_mask])
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 记录损失和预测
                train_loss += loss.item()
                preds = outputs[valid_mask].argmax(dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch.y[valid_mask].cpu().numpy())
                
                # 记录到TensorBoard
                writer.add_scalar('batch_loss', loss.item(), epoch * len(train_loader) + len(train_preds))
            
            except Exception as e:
                print(f"训练批次出错: {e}")
                continue
        
        # 计算训练指标
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        print(f"Epoch {epoch+1}/{args.epoch} (Validation)")
        val_metrics, val_labels, val_preds, val_probs, val_losses = evaluate_model(model, valid_loader, device)
        
        # 更新指标跟踪器
        metrics_tracker.update_metrics(
            epoch,
            {'loss': train_loss, 'acc': train_acc, 'f1': train_f1},
            val_metrics
        )
        
        # 记录到TensorBoard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/accuracy', train_acc, epoch)
        writer.add_scalar('train/f1', train_f1, epoch)
        writer.add_scalar('val/loss', val_metrics['avg_loss'], epoch)
        writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('val/f1', val_metrics['f1'], epoch)
        
        # 打印当前轮次结果
        print(f"Epoch {epoch+1}/{args.epoch}:")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Valid Loss: {val_metrics['avg_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 保存最佳模型
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict()
            print(f"  找到新的最佳模型! F1: {best_val_f1:.4f}")
            
            # 保存最佳模型
            best_model_path = os.path.join(output_dir, f"{run_id}_best_model.pt")
            torch.save(best_model_state, best_model_path)
            print(f"  最佳模型已保存到: {best_model_path}")
    
    # 保存最终模型
    final_model_path = os.path.join(output_dir, f"{run_id}_final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")
    
    # 加载最佳模型进行测试
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 测试阶段
    print("\n开始在测试集上评估最佳模型...")
    test_metrics, test_labels, test_preds, test_probs, test_losses = evaluate_model(model, test_loader, device)
    
    # 更新指标跟踪器
    metrics_tracker.update_test_results(test_metrics)
    metrics_tracker.update_predictions(test_labels, test_preds, test_probs)
    
    # 记录到TensorBoard
    writer.add_scalar('test/accuracy', test_metrics['accuracy'])
    writer.add_scalar('test/f1', test_metrics['f1'])
    writer.add_scalar('test/precision', test_metrics['precision'])
    writer.add_scalar('test/recall', test_metrics['recall'])
    
    # 打印测试结果
    print("\n===== 测试结果 =====")
    print(f"准确率: {test_metrics['accuracy']:.4f}")
    print(f"F1分数: {test_metrics['f1']:.4f}")
    print(f"精确率: {test_metrics['precision']:.4f}")
    print(f"召回率: {test_metrics['recall']:.4f}")
    print(f"平衡准确率: {test_metrics['balanced_accuracy']:.4f}")
    
    # 生成各种评估图表
    if args.generate_plots:
        print("\n生成评估图表...")
        metrics_tracker.generate_report()
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 计算运行时间
    time_end = time.time()
    print(f"总运行时间: {time_end - time_start:.2f} 秒")
    
    return test_metrics

if __name__ == "__main__":
    try:
        print("\n===== SeGA-LLM 模型启动 =====")
        print("使用内置配置，无需命令行参数")
        print(f"模型路径: {CONFIG['model_path']}")
        print(f"数据路径: {CONFIG['dataset_path']}")
        print(f"输出目录: {CONFIG['output_dir']}")
        print("============================\n")
        
        # 运行主函数
        results = main()
        
        # 打印结果摘要
        if results:
            print("\n===== 结果摘要 =====")
            print(f"准确率: {results['accuracy']:.4f}")
            print(f"F1分数: {results['f1']:.4f}")
            print(f"输出保存在: {CONFIG['output_dir']}")
        
        print("\n===== 运行完成 =====")
        
    except ImportError as e:
        print(f"\n错误: 缺少必要的库 - {e}")
        print("请确保已安装所有必要的依赖项:")
        print("pip install torch pytorch-lightning torch-geometric scikit-learn matplotlib seaborn pandas tqdm")
        
    except FileNotFoundError as e:
        print(f"\n错误: 找不到文件或目录 - {e}")
        print("请检查配置中的路径是否正确:")
        print(f"- 模型路径: {CONFIG['model_path']}")
        print(f"- 数据路径: {CONFIG['dataset_path']}")
        
    except torch.cuda.OutOfMemoryError:
        print("\n错误: GPU内存不足")
        print("建议:")
        print("1. 减小批次大小 (CONFIG['batch_size'])")
        print("2. 减小嵌入维度 (CONFIG['dim'])")
        print("3. 使用CPU运行 (设置CONFIG['cuda']为-1)")
        
    except Exception as e:
        import traceback
        print(f"\n运行时错误: {e}")
        traceback.print_exc()
        print("\n尝试调整配置或检查数据格式。") 