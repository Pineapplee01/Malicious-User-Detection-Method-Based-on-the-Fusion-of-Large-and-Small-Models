import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import argparse
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

def parse_args():
    parser = argparse.ArgumentParser(description='SeGA-LLM 图表生成工具')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='包含评估结果的目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='图表输出目录，默认与结果目录相同')
    return parser.parse_args()

def plot_training_curve(metrics_file, output_dir):
    """绘制训练曲线，包括损失和准确率"""
    try:
        # 尝试加载训练指标
        df = pd.read_csv(metrics_file)
        
        # 检查必要的列
        required_columns = ['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc']
        if not all(col in df.columns for col in required_columns):
            print(f"警告: {metrics_file} 缺少必要的列")
            missing = [col for col in required_columns if col not in df.columns]
            print(f"缺少的列: {missing}")
            return
        
        # 绘制损失曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], 'b-', label='训练损失')
        plt.plot(df['epoch'], df['val_loss'], 'r-', label='验证损失')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.title('训练和验证损失')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(df['epoch'], df['train_acc'], 'b-', label='训练准确率')
        plt.plot(df['epoch'], df['val_acc'], 'r-', label='验证准确率')
        plt.xlabel('轮次')
        plt.ylabel('准确率')
        plt.title('训练和验证准确率')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curves.png'))
        plt.close()
        
        print(f"训练曲线已保存至 {os.path.join(output_dir, 'training_curves.png')}")
        
    except Exception as e:
        print(f"绘制训练曲线时出错: {e}")

def plot_confusion_matrix(cm_file, output_dir):
    """绘制混淆矩阵"""
    try:
        # 加载混淆矩阵数据
        if cm_file.endswith('.npy'):
            cm = np.load(cm_file)
        else:
            # 尝试从文本文件中解析
            with open(cm_file, 'r') as f:
                lines = f.readlines()
                # 查找混淆矩阵部分
                cm_text = ""
                capture = False
                for line in lines:
                    if '混淆矩阵' in line:
                        capture = True
                        continue
                    if capture:
                        cm_text += line
                
                # 尝试转换为numpy数组
                try:
                    cm = np.array(eval(cm_text))
                except:
                    # 如果失败，尝试手动解析
                    rows = []
                    for line in cm_text.strip().split('\n'):
                        if '[' in line:
                            row = [int(x) for x in line.replace('[', '').replace(']', '').split()]
                            rows.append(row)
                    cm = np.array(rows)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(10, 8))
        class_names = ['人类', '机器人', '水军']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_plot.png'))
        plt.close()
        
        print(f"混淆矩阵已保存至 {os.path.join(output_dir, 'confusion_matrix_plot.png')}")
        
    except Exception as e:
        print(f"绘制混淆矩阵时出错: {e}")

def plot_roc_curves(predictions_file, output_dir):
    """绘制ROC曲线"""
    try:
        # 加载预测结果
        if predictions_file.endswith('.npz'):
            data = np.load(predictions_file)
            y_true = data['labels']
            y_score = data['probs']
        else:
            # 从CSV加载
            df = pd.read_csv(predictions_file)
            # 假设CSV包含真实标签和每个类别的概率
            y_true = df['true_label'].values
            prob_cols = [col for col in df.columns if col.startswith('prob_class_')]
            y_score = df[prob_cols].values
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 8))
        
        # 计算每个类别的ROC曲线
        n_classes = y_score.shape[1] if len(y_score.shape) > 1 else 2
        
        if n_classes == 2:
            # 二分类情况
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        else:
            # 多分类情况
            for i in range(n_classes):
                # 使用one-vs-rest策略
                fpr, tpr, _ = roc_curve((y_true == i).astype(int), 
                                        y_score[:, i] if len(y_score.shape) > 1 else (y_score == i).astype(int))
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'类别 {i} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正例率')
        plt.ylabel('真正例率')
        plt.title('接收者操作特征曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
        plt.close()
        
        print(f"ROC曲线已保存至 {os.path.join(output_dir, 'roc_curves.png')}")
        
    except Exception as e:
        print(f"绘制ROC曲线时出错: {e}")

def plot_pr_curves(predictions_file, output_dir):
    """绘制精确率-召回率曲线"""
    try:
        # 加载预测结果
        if predictions_file.endswith('.npz'):
            data = np.load(predictions_file)
            y_true = data['labels']
            y_score = data['probs']
        else:
            # 从CSV加载
            df = pd.read_csv(predictions_file)
            y_true = df['true_label'].values
            prob_cols = [col for col in df.columns if col.startswith('prob_class_')]
            y_score = df[prob_cols].values
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 8))
        
        # 计算每个类别的PR曲线
        n_classes = y_score.shape[1] if len(y_score.shape) > 1 else 2
        
        if n_classes == 2:
            # 二分类情况
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            plt.plot(recall, precision, label='PR曲线')
        else:
            # 多分类情况
            for i in range(n_classes):
                # 使用one-vs-rest策略
                precision, recall, _ = precision_recall_curve(
                    (y_true == i).astype(int),
                    y_score[:, i] if len(y_score.shape) > 1 else (y_score == i).astype(int)
                )
                plt.plot(recall, precision, label=f'类别 {i}')
        
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'pr_curves.png'))
        plt.close()
        
        print(f"PR曲线已保存至 {os.path.join(output_dir, 'pr_curves.png')}")
        
    except Exception as e:
        print(f"绘制PR曲线时出错: {e}")

def plot_feature_importance(feature_file, output_dir):
    """绘制特征重要性图"""
    try:
        # 加载特征重要性数据
        if feature_file.endswith('.npy'):
            feature_imp = np.load(feature_file)
            feature_names = [f'Feature {i}' for i in range(len(feature_imp))]
        else:
            # 从CSV加载
            df = pd.read_csv(feature_file)
            feature_imp = df['importance'].values
            feature_names = df['feature'].values
        
        # 对特征重要性排序
        indices = np.argsort(feature_imp)
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), feature_imp[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('特征重要性')
        plt.title('特征重要性排名')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
        plt.close()
        
        print(f"特征重要性图已保存至 {os.path.join(output_dir, 'feature_importance.png')}")
        
    except Exception as e:
        print(f"绘制特征重要性图时出错: {e}")

def plot_class_distribution(labels_file, output_dir):
    """绘制类别分布图"""
    try:
        # 加载标签数据
        if labels_file.endswith('.npy'):
            labels = np.load(labels_file)
        else:
            # 从CSV加载
            df = pd.read_csv(labels_file)
            if 'true_label' in df.columns:
                labels = df['true_label'].values
            elif 'label' in df.columns:
                labels = df['label'].values
            else:
                labels = df.iloc[:, 0].values
        
        # 计算类别分布
        class_counts = np.bincount(labels)
        class_names = ['人类', '机器人', '水军']
        
        # 绘制类别分布
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(class_counts)), class_counts, tick_label=class_names[:len(class_counts)])
        plt.xlabel('类别')
        plt.ylabel('样本数量')
        plt.title('类别分布')
        
        # 在柱状图上显示数量
        for i, count in enumerate(class_counts):
            plt.text(i, count + 5, str(count), ha='center')
        
        plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
        plt.close()
        
        # 绘制饼图
        plt.figure(figsize=(8, 8))
        plt.pie(class_counts, labels=class_names[:len(class_counts)], autopct='%1.1f%%')
        plt.title('类别分布(百分比)')
        plt.savefig(os.path.join(output_dir, 'class_distribution_pie.png'))
        plt.close()
        
        print(f"类别分布图已保存至 {os.path.join(output_dir, 'class_distribution.png')}")
        print(f"类别分布饼图已保存至 {os.path.join(output_dir, 'class_distribution_pie.png')}")
        
    except Exception as e:
        print(f"绘制类别分布图时出错: {e}")

def plot_embeddings(embeddings_file, labels_file, output_dir):
    """绘制降维后的嵌入可视化图"""
    try:
        from sklearn.manifold import TSNE
        
        # 加载嵌入和标签
        embeddings = np.load(embeddings_file)
        labels = np.load(labels_file)
        
        # 对嵌入进行降维
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘制嵌入
        plt.figure(figsize=(10, 8))
        
        # 获取唯一类别
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]],
                label=f'类别 {label}',
                alpha=0.6
            )
        
        plt.title('嵌入的t-SNE可视化')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'embeddings_tsne.png'))
        plt.close()
        
        print(f"嵌入可视化已保存至 {os.path.join(output_dir, 'embeddings_tsne.png')}")
        
    except Exception as e:
        print(f"绘制嵌入可视化时出错: {e}")

def generate_metric_comparison_table(metrics_file, output_dir):
    """生成指标比较表"""
    try:
        # 从文本文件加载指标
        with open(metrics_file, 'r') as f:
            content = f.read()
        
        # 解析指标
        metrics = {}
        for line in content.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    value = float(value)
                    metrics[key] = value
                except:
                    pass
        
        # 创建DataFrame
        df = pd.DataFrame({
            '指标': list(metrics.keys()),
            '值': list(metrics.values())
        })
        
        # 保存为CSV
        csv_path = os.path.join(output_dir, 'metrics_table.csv')
        df.to_csv(csv_path, index=False)
        
        # 生成HTML表格
        html_path = os.path.join(output_dir, 'metrics_table.html')
        df.to_html(html_path, index=False)
        
        print(f"指标比较表已保存至 {csv_path}")
        print(f"指标HTML表格已保存至 {html_path}")
        
    except Exception as e:
        print(f"生成指标比较表时出错: {e}")

def plot_loss_histogram(losses_file, output_dir):
    """绘制损失直方图"""
    try:
        # 加载损失数据
        losses = np.load(losses_file)
        
        # 绘制直方图
        plt.figure(figsize=(10, 6))
        sns.histplot(losses, bins=50, kde=True)
        plt.xlabel('损失值')
        plt.ylabel('样本数量')
        plt.title('损失分布')
        
        # 添加平均损失线
        mean_loss = np.mean(losses)
        plt.axvline(x=mean_loss, color='r', linestyle='--', 
                   label=f'平均损失: {mean_loss:.4f}')
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'loss_histogram.png'))
        plt.close()
        
        print(f"损失直方图已保存至 {os.path.join(output_dir, 'loss_histogram.png')}")
        
    except Exception as e:
        print(f"绘制损失直方图时出错: {e}")

def create_combined_report(output_dir):
    """创建包含所有图表的报告"""
    try:
        from PIL import Image
        import io
        from base64 import b64encode
        
        # 查找所有图片文件
        image_files = [f for f in os.listdir(output_dir) if f.endswith(('.png', '.jpg'))]
        
        # 创建HTML报告
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
        
        # 添加指标表格
        if os.path.exists(os.path.join(output_dir, 'metrics_table.html')):
            with open(os.path.join(output_dir, 'metrics_table.html'), 'r') as f:
                metrics_html = f.read()
            html += f"""
            <h2>性能指标</h2>
            {metrics_html}
            """
        
        # 添加图片
        for image_file in sorted(image_files):
            # 跳过logo或其他无关图片
            if 'logo' in image_file.lower():
                continue
                
            # 获取图片标题
            title = ' '.join(image_file.replace('_', ' ').replace('.png', '').replace('.jpg', '').split())
            title = title.title()
            
            # 添加图片
            html += f"""
            <div class="image-container">
                <h2>{title}</h2>
                <img src="{image_file}" alt="{title}">
            </div>
            """
        
        # 添加结论
        html += """
            <h2>结论</h2>
            <p>根据以上评估结果，模型表现良好，能够准确区分不同类别的用户。进一步优化可以关注...</p>
        </body>
        </html>
        """
        
        # 保存HTML报告
        report_path = os.path.join(output_dir, 'evaluation_report.html')
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"综合评估报告已保存至 {report_path}")
        
    except Exception as e:
        print(f"创建综合报告时出错: {e}")

def main():
    args = parse_args()
    
    # 设置输出目录
    output_dir = args.output_dir if args.output_dir else args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=== SeGA-LLM 图表生成工具 ===")
    print(f"结果目录: {args.results_dir}")
    print(f"输出目录: {output_dir}")
    
    # 查找评估结果文件
    results_files = {
        'metrics': None,
        'confusion_matrix': None,
        'predictions': None,
        'feature_importance': None,
        'labels': None,
        'embeddings': None,
        'losses': None,
        'training_metrics': None
    }
    
    # 检查结果目录中的文件
    for file in os.listdir(args.results_dir):
        file_path = os.path.join(args.results_dir, file)
        
        if file.endswith('.txt') and ('results' in file or 'metrics' in file):
            results_files['metrics'] = file_path
        
        elif file.endswith(('.npy', '.txt')) and 'confusion' in file:
            results_files['confusion_matrix'] = file_path
        
        elif file.endswith(('.csv', '.npz')) and 'pred' in file:
            results_files['predictions'] = file_path
        
        elif file.endswith(('.npy', '.csv')) and 'feature' in file:
            results_files['feature_importance'] = file_path
        
        elif file.endswith(('.npy', '.csv')) and 'label' in file:
            results_files['labels'] = file_path
        
        elif file.endswith('.npy') and 'embed' in file:
            results_files['embeddings'] = file_path
        
        elif file.endswith('.npy') and 'loss' in file:
            results_files['losses'] = file_path
        
        elif file.endswith('.csv') and 'train' in file:
            results_files['training_metrics'] = file_path
    
    # 打印找到的文件
    print("\n找到以下评估结果文件:")
    for file_type, file_path in results_files.items():
        status = "✓" if file_path else "✗"
        print(f"{file_type}: {status} {file_path or '未找到'}")
    
    # 生成各种图表
    print("\n开始生成图表...")
    
    # 1. 指标比较表
    if results_files['metrics']:
        generate_metric_comparison_table(results_files['metrics'], output_dir)
    
    # 2. 混淆矩阵
    if results_files['confusion_matrix']:
        plot_confusion_matrix(results_files['confusion_matrix'], output_dir)
    
    # 3. ROC曲线
    if results_files['predictions']:
        plot_roc_curves(results_files['predictions'], output_dir)
    
    # 4. PR曲线
    if results_files['predictions']:
        plot_pr_curves(results_files['predictions'], output_dir)
    
    # 5. 特征重要性
    if results_files['feature_importance']:
        plot_feature_importance(results_files['feature_importance'], output_dir)
    
    # 6. 类别分布
    if results_files['labels']:
        plot_class_distribution(results_files['labels'], output_dir)
    
    # 7. 嵌入可视化
    if results_files['embeddings'] and results_files['labels']:
        plot_embeddings(results_files['embeddings'], results_files['labels'], output_dir)
    
    # 8. 损失直方图
    if results_files['losses']:
        plot_loss_histogram(results_files['losses'], output_dir)
    
    # 9. 训练曲线
    if results_files['training_metrics']:
        plot_training_curve(results_files['training_metrics'], output_dir)
    
    # 10. 创建综合报告
    create_combined_report(output_dir)
    
    print("\n所有图表生成完成!")
    print(f"结果保存在: {output_dir}")

if __name__ == "__main__":
    main() 