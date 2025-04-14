import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch_geometric
from torch_geometric.loader import NeighborLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import os
import sys
import numpy as np
import random
import json
import time
import math
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm
import globals
from datetime import datetime
from tqdm import tqdm

# 确保目录添加到路径中
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', 'SeGA-main', 'Code'))

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SeGA-main/Code')))

# 导入自定义模块
from SeGA_LLM import SeGA_LLM
from data_loader_llm import load_data_llm

def parse_args():
    parser = argparse.ArgumentParser(description='SeGA-LLM')
    parser.add_argument("--dataset_path", type=str, default="../processed_data",
                        help="处理后数据的路径")
    parser.add_argument('--dim', type=int, default=64,
                        help='embedding dimensions')
    parser.add_argument('--cuda', type=int, default=0,
                        help='cuda device')
    parser.add_argument('--template', type=str, default='l',
                        help='template choice')
    parser.add_argument('--list_num', type=int, default=50,
                        help='number of lists')
    parser.add_argument('--epoch', type=int, default=1,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=8,
                        help='test batch size')
    parser.add_argument('--encoder', type=str, default='SimpleGCN',
                        help='choice of encoder')
    parser.add_argument('--prompt_encoder', type=str, default='roberta',
                        help='choice of prompt encoder')
    parser.add_argument('--sample', action='store_true', default=False,
                        help='if sampling neighbors')
    parser.add_argument('--pretrain', action='store_true', default=False,
                        help='whether to pretrain')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--edge_types', type=str, default="all",
                        help='types of edges')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='dropout probability')
    parser.add_argument('--wd', type=float, default=0.0001,
                        help='weight decay')
    parser.add_argument('--temp', type=float, default=0.05,
                        help='temperature for contrastive loss')
    parser.add_argument('--pretext_task', type=str, default='contrastive',
                        help='pretext task')
    parser.add_argument('--lst', action='store_true', default=False,
                        help='if using list features')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='log directory for tensorboard')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='output directory for saving models')
    parser.add_argument('--llm_model', type=str, default='mistral',
                        help='LLM model type to use (mistral/llama/gemma)')
    parser.add_argument('--llm_enhancement_channel', type=int, default=100,
                        help='LLM enhancement channel dimension')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to load pretrained model checkpoint')
    parser.add_argument('--simplified', action='store_true', default=True,
                        help='Use simplified model without PyG dependencies')
    parser.add_argument('--debug', action='store_true', default=True,
                        help='Run in debug mode with minimal data')
    return parser.parse_args()

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(args, pretrain, finetune, model_path):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化全局变量存储测试结果
    globals.init()
    
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
        model = SeGA_LLM(args, pretrain=False)
    
    # 设置保存最佳模型的回调
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='finetune-{epoch:02d}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max',
        save_top_k=1
    )
    
    # 设置训练器
    trainer = pl.Trainer(
        max_epochs=args.finetune_epochs,
        accelerator="gpu",
        devices=args.devices,
        callbacks=[checkpoint_callback],
        default_root_dir=args.output_dir
    )
    
    # 训练模型
    trainer.fit(model, train_loader, valid_loader)
    
    # 评估模型
    trainer.test(model, test_loader)
    
    # 保存最终结果
    test_results = {
        'acc': globals.acc,
        'f1': globals.f1,
        'precision': globals.precision,
        'recall': globals.recall,
        'bacc': globals.bacc
    }
    
    print("测试结果:")
    print(f"准确率: {test_results['acc']:.4f}")
    print(f"F1分数: {test_results['f1']:.4f}")
    print(f"精确率: {test_results['precision']:.4f}")
    print(f"召回率: {test_results['recall']:.4f}")
    print(f"平衡准确率: {test_results['bacc']:.4f}")
    
    return test_results

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
    
    # 解析参数
    args = parse_args()
    
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
    log_dir = os.path.normpath(os.path.join(os.path.abspath(args.log_dir), run_id))
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
        print(f"边数量={data.edge_index.shape}")
    
    # 预处理边索引
    print("预处理图边索引...")
    data = process_edge_index(data, verbose=True)
    
    # 清理不必要的数据节省内存
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 打印数据大小
    print(f"数据准备完成，节点数量: {data.num_nodes}, 边数量: {data.edge_index.size(1)}")
    
    # 创建模型
    print(f"创建模型: 输入维度={data.x.shape[1]}, 嵌入维度={args.dim}")
    model = SeGA_LLM(in_dim=data.x.shape[1],
                    embed_dim=args.dim,
                    dropout=args.dropout,
                    temp=args.temp,
                    pretext_task=args.pretext_task,
                    pretrain=args.pretrain,
                    device=device,
                    encoder=args.encoder,
                    args=args)
    
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
    
    # 使用内存节省技术
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    # 训练和评估
    try:
        if args.pretrain:
            print("开始预训练...")
            print(f"使用{'列表特征' if args.lst else '仅用户特征'}")
            print(f"使用{args.encoder}编码器")
            pretrain_state = model.pretrain(data=data, epoch=args.epoch, batch_size=args.batch_size,
                                      writer=writer, device=device, embedding_path=os.path.join(output_dir, f"{run_id}_pretrain"),
                                      use_tqdm=True)
            # 保存预训练模型
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_id}_pretrained_model.pt"))
            print(f"预训练模型已保存至 {os.path.join(output_dir, f'{run_id}_pretrained_model.pt')}")
        else:
            print("开始微调...")
            print(f"使用{'列表特征' if args.lst else '仅用户特征'}")
            print(f"使用{args.encoder}编码器")
            finetune_state = model.finetune(data=data, epoch=args.epoch, batch_size=args.batch_size,
                                      writer=writer, device=device, embedding_path=os.path.join(output_dir, f"{run_id}_finetune"),
                                      use_tqdm=True)
            # 保存微调模型
            torch.save(model.state_dict(), os.path.join(output_dir, f"{run_id}_finetuned_model.pt"))
            print(f"微调模型已保存至 {os.path.join(output_dir, f'{run_id}_finetuned_model.pt')}")
    except Exception as e:
        print(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
    
    # 关闭TensorBoard写入器
    writer.close()
    
    # 计算运行时间
    time_end = time.time()
    print(f"总运行时间: {time_end - time_start:.2f} 秒")

if __name__ == "__main__":
    main() 