"""
辅助脚本: 创建LLM特征占位文件
此脚本创建占位LLM特征文件，确保数据加载过程能够正常工作。
"""

import os
import torch
import argparse
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='Create placeholder LLM feature files')
    parser.add_argument('--data_path', type=str, default="../processed_data",
                        help='Path to processed data')
    parser.add_argument('--llm_model', type=str, default="mistral",
                        help='LLM model name (e.g., mistral, llama, gemma)')
    parser.add_argument('--dimensions', type=int, default=100,
                        help='Number of dimensions for LLM features')
    parser.add_argument('--force', action='store_true', 
                        help='Force overwrite existing files')
    parser.add_argument('--all_models', action='store_true',
                        help='Create features for all models (mistral, llama, gemma)')
    return parser.parse_args()

def create_llm_features(data_path, llm_model, dimensions, force=False):
    """为指定的LLM模型创建特征文件"""
    print(f"\n处理 {llm_model} 模型...")
    
    # 加载用户和列表的基本张量以确定大小
    try:
        # 用户特征
        user_cat_features_path = os.path.join(data_path, "user_cat_properties_tensor.pt")
        if os.path.exists(user_cat_features_path):
            user_cat_features = torch.load(user_cat_features_path, map_location='cpu')
            user_count = user_cat_features.shape[0]
            print(f"用户数量: {user_count}")
            
            # 创建用户LLM特征文件
            user_llm_features_path = os.path.join(data_path, f"user_llm_features_{llm_model}.pt")
            if os.path.exists(user_llm_features_path) and not force:
                print(f"用户LLM特征文件已存在: {user_llm_features_path} (使用--force覆盖)")
            else:
                print(f"创建用户LLM特征文件: {user_llm_features_path}")
                user_llm_features = torch.zeros((user_count, dimensions), dtype=torch.float)
                torch.save(user_llm_features, user_llm_features_path)
                print(f"已保存用户LLM特征，形状: {user_llm_features.shape}")
        
        # 列表特征
        list_cat_features_path = os.path.join(data_path, "list_cat_properties_tensor.pt")
        if os.path.exists(list_cat_features_path):
            list_cat_features = torch.load(list_cat_features_path, map_location='cpu')
            list_count = list_cat_features.shape[0]
            print(f"列表数量: {list_count}")
            
            # 创建列表LLM特征文件
            list_llm_features_path = os.path.join(data_path, f"list_llm_features_{llm_model}.pt")
            if os.path.exists(list_llm_features_path) and not force:
                print(f"列表LLM特征文件已存在: {list_llm_features_path} (使用--force覆盖)")
            else:
                print(f"创建列表LLM特征文件: {list_llm_features_path}")
                list_llm_features = torch.zeros((list_count, dimensions), dtype=torch.float)
                torch.save(list_llm_features, list_llm_features_path)
                print(f"已保存列表LLM特征，形状: {list_llm_features.shape}")
        
        print(f"{llm_model} 模型处理完成!")
        return True
        
    except Exception as e:
        print(f"处理 {llm_model} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 解析参数
    args = parse_args()
    
    # 将相对路径转换为绝对路径
    data_path = os.path.abspath(args.data_path)
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        print(f"错误: 数据路径 '{data_path}' 不存在")
        return
    
    print(f"数据路径: {data_path}")
    print(f"特征维度: {args.dimensions}")
    
    # 创建要处理的模型列表
    models_to_process = ["mistral", "llama", "gemma"] if args.all_models else [args.llm_model]
    print(f"将处理以下模型: {', '.join(models_to_process)}")
    
    # 为每个模型创建特征
    success_count = 0
    for model in models_to_process:
        if create_llm_features(data_path, model, args.dimensions, args.force):
            success_count += 1
    
    print(f"\n总结: 成功处理 {success_count}/{len(models_to_process)} 个模型")

if __name__ == "__main__":
    main() 