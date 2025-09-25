"""
极简直接测试脚本
不使用复杂的数据加载函数，直接加载必要文件并评估模型
"""

import os
import torch
import numpy as np
from pathlib import Path
import time

def main():
    print("\n===== SeGA-LLM 极简测试脚本 =====")
    start_time = time.time()
    
    # 1. 设置路径
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    base_dir = current_dir.parent
    data_dir = base_dir / "processed_data"
    model_path = current_dir / "checkpoints" / "20250410-171222_finetune_best_model.pt"
    output_dir = base_dir / "evaluation_results"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 检查文件
    print(f"检查文件路径...")
    print(f"当前目录: {current_dir}")
    print(f"数据目录: {data_dir}")
    print(f"模型文件: {model_path}")
    print(f"输出目录: {output_dir}")
    
    if not model_path.exists():
        print(f"错误: 模型文件不存在 - {model_path}")
        return
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在 - {data_dir}")
        return
    
    # 3. 加载模型
    print(f"\n正在加载模型: {model_path}")
    try:
        # 导入所需模块
        from SeGA_LLM import SeGA_LLM
        
        # 设置设备
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 创建模型所需的args对象
        args = type('Args', (), {
            'lr': 0.001,
            'wd': 0.0001,
            'batch_size': 8,
            'lst': True,
            'list_num': 50,
            'llm_enhancement_channel': 103,
            'prompt_encoder': 'roberta',
            'template': 'l',
            'dataset_path': str(data_dir)
        })()
        
        # 加载模型权重
        state_dict = torch.load(model_path, map_location=device)
        
        # 初始化模型 (使用正确的参数)
        model = SeGA_LLM(
            in_dim=768,          # 通用输入维度
            embed_dim=16,        # 使用小尺寸嵌入
            dropout=0.3,
            temp=0.05,
            pretext_task="contrastive",
            pretrain=False,
            device=device,
            encoder="SimpleGCN",
            args=args
        )
        
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        print("模型加载成功")
        
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
        
    # 4. 加载测试数据
    print(f"\n正在加载测试数据...")
    try:
        # 加载测试索引
        test_idx_path = data_dir / "finetune_test_idx.pt"
        if not test_idx_path.exists():
            print(f"错误: 测试索引文件不存在 - {test_idx_path}")
            return
            
        test_idx = torch.load(test_idx_path, map_location='cpu')
        print(f"测试样本数量: {len(test_idx)}")
        
        # 加载特征和标签
        print("加载特征和标签...")
        features_paths = {
            "user_cat": data_dir / "user_cat_properties_tensor.pt",
            "user_prop": data_dir / "user_num_properties_tensor.pt",
            "user_tweet": data_dir / "user_tweets_tensor.pt",
            "user_des": data_dir / "user_des_tensor.pt",
            "user_llm": data_dir / "user_llm_features_mistral.pt"
        }
        
        # 检查文件是否存在
        for name, path in features_paths.items():
            if not path.exists():
                print(f"错误: {name} 文件不存在 - {path}")
                return
        
        # 加载特征
        features = {}
        for name, path in features_paths.items():
            features[name] = torch.load(path, map_location='cpu')
            print(f"已加载 {name}: {features[name].shape}")
        
        # 加载标签
        labels_path = data_dir / "finetune_label.pt"
        if not labels_path.exists():
            print(f"错误: 标签文件不存在 - {labels_path}")
            return
            
        labels = torch.load(labels_path, map_location='cpu')
        print(f"标签形状: {labels.shape}")
        
        # 合并特征
        user_x = torch.cat([
            features["user_cat"],
            features["user_prop"],
            features["user_tweet"],
            features["user_des"],
            features["user_llm"]
        ], dim=1)
        print(f"合并后特征形状: {user_x.shape}")
        
    except Exception as e:
        print(f"加载测试数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 评估模型
    print(f"\n开始评估模型...")
    try:
        batch_size = 8
        all_preds = []
        all_labels = []
        
        # 创建一个简单的批次对象
        class SimpleDataBatch:
            def __init__(self, x, n_id, num_nodes, y=None):
                self.x = x
                self.n_id = n_id
                self.num_nodes = num_nodes
                # 添加空的edge_index属性
                self.edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                # 添加标签属性
                self.y = y
                # 添加train_mask属性
                self.train_mask = torch.ones(num_nodes, dtype=torch.bool, device=device)
                self.val_mask = None
                self.test_mask = None
        
        # 添加forward方法到模型类
        def forward_function(self, x):
            # 创建模拟批次
            batch = SimpleDataBatch(
                x=x,
                n_id=torch.arange(x.size(0), device=x.device),
                num_nodes=x.size(0)
            )
            
            # 复制training_step中特征处理的逻辑
            features = torch.zeros(batch.num_nodes, self.embed_dim, dtype=torch.float32, device=x.device)
            
            # 处理用户节点特征
            user_mask = torch.ones(batch.num_nodes, dtype=torch.bool, device=x.device)
            if user_mask.sum() > 0:
                user_x = batch.x.clone()
                
                # 动态确定特征维度
                llm_start = self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel
                
                # 检查特征划分是否有效
                if user_x.shape[1] < llm_start:
                    raise ValueError(f"用户特征维度不足: {user_x.shape[1]}, 需要至少 {llm_start}")
                
                user_cat_features = user_x[:, :self.list_cat_num]
                user_prop_features = user_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                user_tweet_features = user_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                user_des_features = user_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: llm_start]
                user_llm_features = user_x[:, llm_start:]
                
                # 用户特征处理
                user_features_numeric = self.numeric_layer(user_prop_features)
                user_features_bool = self.cat_layer(user_cat_features)
                user_features_tweet = self.tweet_layer(user_tweet_features)
                user_features_des = self.des_layer(user_des_features)
                user_features_llm = self.llm_layer(user_llm_features)
                
                user_features = torch.cat((
                    user_features_numeric, 
                    user_features_bool, 
                    user_features_tweet, 
                    user_features_des, 
                    user_features_llm
                ), dim=1)
                
                user_features = self.hidden_layer(user_features)
                features[user_mask] = user_features
            
            # 图处理 (假设GNN层正确设置)
            if hasattr(self, 'gnn') and self.gnn is not None:
                features = self.gnn(features, batch.edge_index)
            
            # 分类层
            features = self.out_layer(features)
            logits = self.classifier(features)
            
            return logits
        
        # 动态添加forward方法到模型
        import types
        model.forward = types.MethodType(forward_function, model)
        
        with torch.no_grad():
            for i in range(0, len(test_idx), batch_size):
                # 获取当前批次索引
                batch_idx = test_idx[i:i+batch_size]
                
                # 获取特征和标签
                batch_features = user_x[batch_idx]
                batch_labels = labels[batch_idx]
                
                # 创建批次对象
                batch = SimpleDataBatch(
                    x=batch_features, 
                    n_id=torch.arange(len(batch_idx)), 
                    num_nodes=len(batch_idx),
                    y=batch_labels
                )
                
                # 移动到设备
                batch.x = batch.x.to(device)
                batch.y = batch.y.to(device)
                
                # 前向传播 - 直接调用模型获取logits而不是使用test_step
                logits = model(batch.x)
                
                # 获取预测
                preds = torch.argmax(logits, dim=1)
                
                # 收集结果
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
                # 打印进度
                if (i // batch_size) % 10 == 0:
                    print(f"已处理 {i+len(batch_idx)}/{len(test_idx)} 个样本")
        
        # 6. 计算指标
        print("\n计算评估指标...")
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        # 添加混淆矩阵可视化
        print("生成混淆矩阵可视化...")
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 配置支持中文显示
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
            plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True)
            plt.xlabel('预测标签')
            plt.ylabel('真实标签')
            plt.title('混淆矩阵')
            
            # 保存混淆矩阵图
            cm_image_path = output_dir / "confusion_matrix.png"
            plt.savefig(cm_image_path)
            plt.close()
            print(f"混淆矩阵图已保存到: {cm_image_path}")
            
            # 生成准确率、精确率、召回率和F1分数的柱状图
            metrics = {
                '准确率': accuracy,
                '精确率': precision,
                '召回率': recall,
                'F1分数': f1
            }
            
            # 如果中文显示仍有问题，可以使用英文标签作为备选
            english_metrics = {
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1
            }
            
            # 尝试使用中文标签
            try:
                plt.figure(figsize=(10, 6))
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                bars = plt.bar(metrics.keys(), metrics.values(), color=colors)
                
                # 在柱状图上添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom')
                
                plt.ylim(0, 1.0)
                plt.title('评估指标')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 保存指标图
                metrics_image_path = output_dir / "metrics.png"
                plt.savefig(metrics_image_path)
                plt.close()
                print(f"评估指标图已保存到: {metrics_image_path}")
            except Exception as font_error:
                # 如果中文显示失败，使用英文标签
                print(f"中文标签显示失败，使用英文标签: {font_error}")
                plt.figure(figsize=(10, 6))
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                bars = plt.bar(english_metrics.keys(), english_metrics.values(), color=colors)
                
                # 在柱状图上添加数值标签
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{height:.4f}', ha='center', va='bottom')
                
                plt.ylim(0, 1.0)
                plt.title('Performance Metrics')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # 保存指标图
                metrics_image_path = output_dir / "metrics_english.png"
                plt.savefig(metrics_image_path)
                plt.close()
                print(f"英文评估指标图已保存到: {metrics_image_path}")
                
        except Exception as e:
            print(f"生成图表时出错: {e}")
            print("继续执行，但不生成图表...")
        
        # 7. 输出结果
        print("\n===== 评估结果 =====")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        # 8. 保存结果
        result_file = output_dir / "direct_test_results.txt"
        with open(result_file, "w") as f:
            f.write("===== SeGA-LLM 直接测试评估结果 =====\n\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"精确率: {precision:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n\n")
            f.write(f"混淆矩阵:\n{cm}\n")
        
        print(f"\n结果已保存到: {result_file}")
            
    except Exception as e:
        print(f"评估模型失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 计算总耗时
    elapsed_time = time.time() - start_time
    print(f"\n评估完成，总耗时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 