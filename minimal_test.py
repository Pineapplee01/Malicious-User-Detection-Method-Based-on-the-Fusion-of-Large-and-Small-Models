import os
import torch
import argparse

# 从SeGA_LLM模块导入模型类
from SeGA_LLM import SeGA_LLM

def parse_args():
    parser = argparse.ArgumentParser(description="最小化模型测试脚本")
    
    # 模型路径参数
    parser.add_argument("--model_path", type=str, default="C:/Users/10900/Desktop/1/1111/node/恶意用户感知/output/20250410-171222_finetune_best_model.pt",
                        help="模型文件路径")
    
    # 模型参数
    parser.add_argument("--dim", type=int, default=16,
                        help="嵌入维度")
    parser.add_argument("--llm_enhancement_channel", type=int, default=103,
                        help="LLM特征通道维度")
    parser.add_argument("--encoder", type=str, default="SimpleGCN",
                        help="编码器类型")
    parser.add_argument("--prompt_encoder", type=str, default="roberta",
                        help="提示编码器类型")
    parser.add_argument("--pretext_task", type=str, default="contrastive",
                        help="预训练任务类型")
    parser.add_argument("--temp", type=float, default=0.05,
                        help="对比损失的温度")
    parser.add_argument("--lst", action="store_true", default=True,
                        help="是否使用列表特征")
    parser.add_argument("--list_num", type=int, default=200000,
                        help="列表数量")
    
    # 添加缺少的参数
    parser.add_argument("--wd", type=float, default=0.0001,
                        help="权重衰减")
    parser.add_argument("--sample", action="store_true", default=False,
                        help="是否采样邻居")
    parser.add_argument("--pretrain", action="store_true", default=False,
                        help="是否预训练")
    parser.add_argument("--simplified", action="store_true", default=True,
                        help="是否使用简化模型")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="批次大小")
    
    # 其他必要参数
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--epoch", type=int, default=1, help="训练轮数")
    parser.add_argument("--edge_types", type=str, default="ff", help="边类型")
    parser.add_argument("--llm_model", type=str, default="mistral", help="LLM模型")
    parser.add_argument("--template", type=str, default="s", help="模板类型")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA设备编号")
    
    # 从检查点发现的特征维度相关参数
    parser.add_argument("--numeric_dim", type=int, default=4, help="数值特征维度")
    parser.add_argument("--cat_dim", type=int, default=1, help="分类特征维度")
    parser.add_argument("--tweet_dim", type=int, default=768, help="推文特征维度")
    parser.add_argument("--des_dim", type=int, default=768, help="描述特征维度")
    
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 准备设备
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        print(f"当前工作目录: {os.getcwd()}")
        
        # 尝试查找替代模型文件
        output_dir = os.path.dirname(args.model_path)
        if os.path.exists(output_dir):
            model_files = [f for f in os.listdir(output_dir) if f.endswith('.pt')]
            if model_files:
                print(f"找到以下模型文件: {model_files}")
                alt_model_path = os.path.join(output_dir, model_files[0])
                print(f"使用替代模型: {alt_model_path}")
                args.model_path = alt_model_path
            else:
                print(f"在 {output_dir} 中未找到任何模型文件")
                return
        else:
            print(f"输出目录不存在: {output_dir}")
            return
    
    try:
        # 加载模型检查点
        print(f"正在加载模型检查点: {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # 打印检查点键，用于调试
        print("\n检查点包含以下键:")
        for key in checkpoint.keys():
            shape_info = ""
            if isinstance(checkpoint[key], torch.Tensor):
                shape_info = f", 形状: {checkpoint[key].shape}"
            print(f"- {key}{shape_info}")
        
        # 检测LLM特征维度
        if 'llm_layer.weight' in checkpoint:
            llm_channel = checkpoint['llm_layer.weight'].shape[1]
            print(f"\n从检查点检测到LLM特征通道维度: {llm_channel}")
            args.llm_enhancement_channel = llm_channel
            
        # 检测嵌入维度
        if 'hidden_layer.weight' in checkpoint:
            hidden_dim = checkpoint['hidden_layer.weight'].shape[0]
            print(f"从检查点检测到嵌入维度: {hidden_dim}")
            args.dim = hidden_dim
            
        # 从检查点更新特征维度
        if 'numeric_layer.weight' in checkpoint:
            args.numeric_dim = checkpoint['numeric_layer.weight'].shape[1]
            print(f"数值特征维度: {args.numeric_dim}")
            
        if 'cat_layer.weight' in checkpoint:
            args.cat_dim = checkpoint['cat_layer.weight'].shape[1]
            print(f"分类特征维度: {args.cat_dim}")
            
        if 'tweet_layer.weight' in checkpoint:
            args.tweet_dim = checkpoint['tweet_layer.weight'].shape[1]
            print(f"推文特征维度: {args.tweet_dim}")
            
        if 'des_layer.weight' in checkpoint:
            args.des_dim = checkpoint['des_layer.weight'].shape[1]
            print(f"描述特征维度: {args.des_dim}")
            
        # 计算总输入维度
        in_dim = args.numeric_dim + args.cat_dim + args.tweet_dim + args.des_dim + args.llm_enhancement_channel
        
        # 创建空模型
        print("\n使用以下参数创建模型:")
        print(f"嵌入维度: {args.dim}")
        print(f"LLM特征通道维度: {args.llm_enhancement_channel}")
        print(f"编码器: {args.encoder}")
        print(f"输入维度: {in_dim}")
        print(f"批次大小: {args.batch_size}")
        
        # 创建模型实例
        model = SeGA_LLM(
            in_dim=in_dim,  # 使用计算出的总输入维度
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
        print("\n尝试加载模型权重...")
        model.load_state_dict(checkpoint)
        print("模型权重加载成功!")
        
        # 将模型移至设备
        model = model.to(device)
        model.eval()
        
        print("\n模型结构:")
        print(model)
        
        print("\n测试成功完成! 模型可以正确加载")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 