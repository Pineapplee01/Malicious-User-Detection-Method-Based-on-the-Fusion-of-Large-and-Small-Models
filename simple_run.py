"""
简化版SeGA-LLM模型运行脚本
该脚本直接预处理所需文件并运行模型，避免复杂的参数传递问题
"""

import os
import sys
import torch
import traceback
from pathlib import Path
import time

def prepare_llm_features(data_path, model_name="mistral", dim=100):
    """
    准备LLM特征文件
    """
    print(f"\n准备{model_name}模型的LLM特征文件...")
    
    try:
        # 确保路径存在
        data_path = Path(data_path)
        if not data_path.exists():
            print(f"错误: 数据路径 {data_path} 不存在")
            return False
            
        # 加载用户特征以获取形状
        user_cat_path = data_path / "user_cat_properties_tensor.pt"
        if user_cat_path.exists():
            user_cat = torch.load(user_cat_path, map_location='cpu')
            user_shape = user_cat.shape[0]
            
            # 创建用户LLM特征
            user_llm_path = data_path / f"user_llm_features_{model_name}.pt"
            if not user_llm_path.exists():
                print(f"创建用户LLM特征文件: {user_llm_path}")
                user_llm = torch.zeros((user_shape, dim), dtype=torch.float)
                torch.save(user_llm, user_llm_path)
            else:
                print(f"用户LLM特征文件已存在: {user_llm_path}")
                
            # 加载列表特征
            list_cat_path = data_path / "list_cat_properties_tensor.pt"
            if list_cat_path.exists():
                list_cat = torch.load(list_cat_path, map_location='cpu')
                list_shape = list_cat.shape[0]
                
                # 创建列表LLM特征
                list_llm_path = data_path / f"list_llm_features_{model_name}.pt"
                if not list_llm_path.exists():
                    print(f"创建列表LLM特征文件: {list_llm_path}")
                    list_llm = torch.zeros((list_shape, dim), dtype=torch.float)
                    torch.save(list_llm, list_llm_path)
                else:
                    print(f"列表LLM特征文件已存在: {list_llm_path}")
            else:
                print(f"警告: 列表特征文件 {list_cat_path} 不存在")
        else:
            print(f"错误: 用户特征文件 {user_cat_path} 不存在")
            return False
            
        print("LLM特征文件准备完成!")
        return True
        
    except Exception as e:
        print(f"准备LLM特征文件时出错: {e}")
        traceback.print_exc()
        return False

def main():
    """
    主函数，简化版的模型运行入口
    """
    start_time = time.time()
    
    print("\n===== SeGA-LLM 简易运行脚本 =====")
    
    # 设置路径
    current_dir = Path(__file__).parent.absolute()
    base_dir = current_dir.parent
    data_dir = base_dir / "processed_data"
    model_path = current_dir / "checkpoints" / "20250410-171222_finetune_best_model.pt"
    output_dir = base_dir / "evaluation_results"
    
    # 输出路径信息
    print(f"当前目录: {current_dir}")
    print(f"数据目录: {data_dir}")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {output_dir}")
    
    # 1. 准备LLM特征文件
    if not prepare_llm_features(data_dir):
        print("准备LLM特征文件失败，退出")
        return
    
    # 2. 导入必要的模块
    try:
        print("\n导入必要模块...")
        # 修正：SeGA-LLM目录不是一个Python包，直接使用相对导入
        current_directory = os.path.dirname(os.path.abspath(__file__))
        if current_directory not in sys.path:
            sys.path.append(current_directory)
            
        # 直接从当前目录导入模块
        from SeGA_LLM import SeGA_LLM
        from data_loader_llm import load_data_llm
        print("模块导入成功")
    except ImportError as e:
        print(f"导入模块失败: {e}")
        traceback.print_exc()
        return
    
    # 3. 准备参数
    print("\n准备模型参数...")
    args = type('Args', (), {
        'dataset_path': str(data_dir),
        'model_path': str(model_path),
        'output_dir': str(output_dir),
        'dim': 16,
        'encoder': 'SimpleGCN',
        'prompt_encoder': 'roberta',
        'llm_model': 'mistral',
        'batch_size': 8,
        'test_batch_size': 8,
        'epoch': 1,
        'lr': 0.001,
        'dropout': 0.3,
        'cuda': 0,
        'seed': 42,
        'simplified': True,
        'debug': True,
        'generate_plots': True,
        'llm_enhancement_channel': 100,
        'lst': True,
        'pretext_task': 'contrastive',
        'template': 'l',
        'edge_types': 'all',
        'wd': 0.0001,
        'temp': 0.05,
        'log_dir': './runs',
        'num_workers': 0,
        'pretrain': False,
        'sample': False,
        'list_num': 50
    })()
    
    # 4. 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 5. 设置设备
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 6. 加载数据
    try:
        print("\n加载数据...")
        pretrain_data, finetune_data = load_data_llm(args)
        print(f"数据加载成功: {len(finetune_data.finetune_test_idx)} 个测试样本")
    except Exception as e:
        print(f"加载数据失败: {e}")
        traceback.print_exc()
        return
    
    # 7. 初始化模型
    try:
        print("\n初始化模型...")
        model = SeGA_LLM(
            in_dim=finetune_data.x.shape[1],
            embed_dim=args.dim,
            out_dim=2,
            dropout=args.dropout,
            encoder=args.encoder,
            prompt_encoder=args.prompt_encoder,
            lr=args.lr
        )
        
        # 8. 加载模型权重
        if model_path.exists():
            print(f"加载模型权重: {model_path}")
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            model = model.to(device)
            model.eval()
            print("模型加载成功")
        else:
            print(f"错误: 模型文件 {model_path} 不存在")
            return
    except Exception as e:
        print(f"初始化或加载模型失败: {e}")
        traceback.print_exc()
        return
    
    # 9. 创建简易数据加载器
    class SimpleBatchLoader:
        def __init__(self, data, batch_size=args.batch_size):
            self.data = data
            self.batch_size = batch_size
            self.test_indices = data.finetune_test_idx
            
        def __iter__(self):
            for i in range(0, len(self.test_indices), self.batch_size):
                batch_indices = self.test_indices[i:min(i + self.batch_size, len(self.test_indices))]
                
                # 创建批次对象
                batch = type('BatchObject', (), {})()
                batch.user_node_feats = self.data.x[batch_indices]
                batch.user_node_masks = torch.ones(len(batch_indices), dtype=torch.bool)
                batch.list_node_feats = None
                batch.list_node_masks = None
                batch.user_edge_index = torch.zeros((2, 0), dtype=torch.long)
                batch.list_edge_index = torch.zeros((2, 0), dtype=torch.long)
                batch.y = self.data.y[batch_indices]
                
                # 移动到设备
                batch.to = lambda device: batch
                yield batch
                
        def __len__(self):
            return (len(self.test_indices) + self.batch_size - 1) // self.batch_size
    
    # 创建数据加载器
    test_loader = SimpleBatchLoader(finetune_data)
    
    # 10. 评估模型
    try:
        print("\n开始评估模型...")
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                # 将数据移动到设备
                batch.user_node_feats = batch.user_node_feats.to(device)
                batch.user_node_masks = batch.user_node_masks.to(device)
                batch.y = batch.y.to(device)
                
                # 前向传播
                outputs = model(
                    user_node_feats=batch.user_node_feats,
                    user_node_masks=batch.user_node_masks,
                    list_node_feats=batch.list_node_feats,
                    list_node_masks=batch.list_node_masks,
                    user_edge_index=batch.user_edge_index,
                    list_edge_index=batch.list_edge_index
                )
                
                # 获取预测
                preds = torch.argmax(outputs, dim=1)
                
                # 收集结果
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch.y.cpu().numpy())
        
        # 计算指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        cm = confusion_matrix(all_labels, all_preds)
        
        # 打印结果
        print("\n===== 评估结果 =====")
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"混淆矩阵:\n{cm}")
        
        # 保存结果
        result_file = output_dir / "evaluation_results.txt"
        with open(result_file, "w") as f:
            f.write("===== SeGA-LLM 评估结果 =====\n\n")
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"精确率: {precision:.4f}\n")
            f.write(f"召回率: {recall:.4f}\n")
            f.write(f"F1分数: {f1:.4f}\n\n")
            f.write(f"混淆矩阵:\n{cm}\n")
        
        print(f"\n结果已保存到: {result_file}")
        
    except Exception as e:
        print(f"评估模型失败: {e}")
        traceback.print_exc()
        return
    
    elapsed_time = time.time() - start_time
    print(f"\n运行完成，用时: {elapsed_time:.2f} 秒")

if __name__ == "__main__":
    main() 