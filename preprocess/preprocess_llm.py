import pandas as pd
import numpy as np
import torch
import os
import sys
import json
from tqdm import tqdm
import argparse

# 添加路径
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
sys.path.append(os.path.join(script_dir, '..', 'SeGA-main', 'Code'))
sys.path.append(os.path.join(script_dir, '..', 'botsay-main'))

from llm_feature_extractor import LLMFeatureExtractor

def parse_args():
    parser = argparse.ArgumentParser(description="LLM特征增强预处理")
    parser.add_argument("--llm_model", type=str, default="mistral", 
                        choices=["mistral", "llama2_7b", "llama2_13b", "llama2_70b", "chatgpt"],
                        help="使用的LLM模型")
    parser.add_argument("--input_path", type=str, 
                        default="/home/TwiBot-22/TwiBot-22/sample_100001/",
                        help="输入数据路径")
    parser.add_argument("--output_path", type=str, 
                        default="./processed_data/",
                        help="输出数据路径")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU设备ID")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 初始化LLM特征提取器
    print(f"初始化LLM特征提取器 ({args.llm_model})...")
    feature_extractor = LLMFeatureExtractor(
        llm_model_name=args.llm_model,
        device=args.device
    )
    
    # 读取用户和列表数据
    print("读取用户和列表数据...")
    user_file = os.path.join(args.input_path, 'user.json')
    list_file = os.path.join(args.input_path, 'list.json')
    
    if not os.path.exists(user_file) or not os.path.exists(list_file):
        print(f"错误: 未找到用户或列表数据文件")
        print(f"用户文件: {user_file}")
        print(f"列表文件: {list_file}")
        return
    
    user_data = pd.read_json(user_file)
    list_data = pd.read_json(list_file)
    
    # 加载用户推文
    print("加载用户推文数据...")
    user_tweets = {}
    
    # 检查推文数据是否存在
    tweets_file = os.path.join(args.output_path, 'uid_tweet.json')
    if os.path.exists(tweets_file):
        try:
            with open(tweets_file, 'r') as f:
                user_tweets = json.load(f)
        except Exception as e:
            print(f"读取推文数据失败: {e}")
            user_tweets = {i: [] for i in range(len(user_data))}
    else:
        print(f"推文文件不存在，将使用空推文列表")
        user_tweets = {i: [] for i in range(len(user_data))}
    
    # 检查列表推文数据是否存在
    list_tweets_file = os.path.join(args.output_path, 'lid_tweet.json')
    if os.path.exists(list_tweets_file):
        try:
            with open(list_tweets_file, 'r') as f:
                list_tweets = json.load(f)
        except Exception as e:
            print(f"读取列表推文数据失败: {e}")
            list_tweets = {i: [] for i in range(len(list_data))}
    else:
        print(f"列表推文文件不存在，将使用空推文列表")
        list_tweets = {i: [] for i in range(len(list_data))}
    
    # 处理用户数据
    print("处理用户描述并生成LLM增强特征...")
    user_llm_features = []
    
    for idx in tqdm(range(len(user_data))):
        # 获取用户描述
        description = user_data.loc[idx, 'description']
        
        # 获取用户推文
        user_tweet_list = user_tweets.get(str(idx), [])
        if not user_tweet_list:
            user_tweet_list = []
        
        # 提取LLM增强特征
        llm_features = feature_extractor.extract_llm_enhanced_features(description, user_tweet_list)
        user_llm_features.append(llm_features)
    
    # 转换为张量并保存
    user_llm_tensor = torch.stack(user_llm_features)
    print(f"用户LLM特征形状: {user_llm_tensor.shape}")
    
    # 保存用户LLM特征
    output_user_file = os.path.join(args.output_path, f'user_llm_features_{args.llm_model}.pt')
    torch.save(user_llm_tensor, output_user_file)
    print(f"用户LLM特征已保存至: {output_user_file}")
    
    # 处理列表数据
    print("处理列表描述并生成LLM增强特征...")
    list_llm_features = []
    
    for idx in tqdm(range(len(list_data))):
        # 获取列表描述
        description = list_data.loc[idx, 'description']
        
        # 获取列表推文
        list_tweet_list = list_tweets.get(str(idx), [])
        if not list_tweet_list:
            list_tweet_list = []
        
        # 提取LLM增强特征
        llm_features = feature_extractor.extract_llm_enhanced_features(description, list_tweet_list)
        list_llm_features.append(llm_features)
    
    # 转换为张量并保存
    list_llm_tensor = torch.stack(list_llm_features)
    print(f"列表LLM特征形状: {list_llm_tensor.shape}")
    
    # 保存列表LLM特征
    output_list_file = os.path.join(args.output_path, f'list_llm_features_{args.llm_model}.pt')
    torch.save(list_llm_tensor, output_list_file)
    print(f"列表LLM特征已保存至: {output_list_file}")
    
    # 清理
    feature_extractor.close()
    print("预处理完成!")

if __name__ == "__main__":
    main() 