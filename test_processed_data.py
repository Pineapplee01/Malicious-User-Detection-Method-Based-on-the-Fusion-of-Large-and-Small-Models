import os
import torch
from data_loader_llm import load_data_llm
from argparse import Namespace

def test_processed_data():
    # 模拟命令行参数
    args = Namespace(
        dataset_path="../processed_data",  # 指定 processed_data 文件夹路径
        llm_model="mistral",              # 使用的 LLM 模型
        edge_types="all",                 # 边类型
        llm_enhancement_channel=100,     # LLM 增强特征维度
        lst=False                         # 是否使用列表特征
    )

    print("开始测试 processed_data 文件夹...")
    try:
        # 加载数据
        pretrain_data, finetune_data = load_data_llm(args)
        print("数据加载成功！")

        # 打印预训练数据的信息
        print("预训练数据:")
        print(f"节点数量: {pretrain_data.num_nodes}")
        print(f"特征维度: {pretrain_data.x.shape}")
        print(f"边数量: {pretrain_data.edge_index.shape[1]}")

        # 打印微调数据的信息
        print("微调数据:")
        print(f"节点数量: {finetune_data.num_nodes}")
        print(f"特征维度: {finetune_data.x.shape}")
        print(f"边数量: {finetune_data.edge_index.shape[1]}")
            
        print("processed_data 测试通过！")
    except Exception as e:
        print(f"processed_data 测试失败: {e}")

if __name__ == "__main__":
    test_processed_data()