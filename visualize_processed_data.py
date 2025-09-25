import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader_llm import load_data_llm
from argparse import Namespace

def visualize_processed_data(dataset_path, llm_model="mistral"):
    # Simulate command-line arguments
    args = Namespace(
        dataset_path=dataset_path,
        llm_model=llm_model,
        edge_types="all",
        llm_enhancement_channel=100,
        lst=False
    )

    print("Loading processed_data...")
    try:
        pretrain_data, finetune_data = load_data_llm(args)
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    # Visualize node feature distribution
    def plot_feature_distribution(data, title):
        feature_sums = data.x.sum(dim=1).numpy()
        plt.figure(figsize=(10, 6))
        sns.histplot(feature_sums, bins=50, kde=True, color='blue', alpha=0.7)
        plt.title(title)
        plt.xlabel("Sum of Features")
        plt.ylabel("Number of Nodes")
        plt.grid(True)
        plt.show()

    # Visualize edge count distribution
    def plot_edge_distribution(data, title):
        edge_counts = torch.bincount(data.edge_index[0])
        plt.figure(figsize=(10, 6))
        sns.histplot(edge_counts.numpy(), bins=50, kde=False, color='green', alpha=0.7)
        plt.title(title)
        plt.xlabel("Number of Edges per Node")
        plt.ylabel("Number of Nodes")
        plt.grid(True)
        plt.show()

    # Visualize label distribution
    def plot_label_distribution(data, title):
        labels = data.y.numpy()
        plt.figure(figsize=(8, 5))
        sns.countplot(x=labels, palette="viridis")
        plt.title(title)
        plt.xlabel("Label Categories")
        plt.ylabel("Number of Nodes")
        plt.grid(True)

        # 设置 x 轴刻度间隔为每 10 个
        max_label = max(labels)
        plt.xticks(ticks=range(0, max_label + 1, 10))  # 每 10 个标签显示一个刻度

        plt.show()

    # Visualize pretraining data
    print("Visualizing pretraining data...")
    plot_feature_distribution(pretrain_data, "Node Feature Distribution (Pretraining Data)")
    plot_edge_distribution(pretrain_data, "Edge Count Distribution (Pretraining Data)")
    plot_label_distribution(pretrain_data, "Label Distribution (Pretraining Data)")

    # Visualize fine-tuning data
    print("Visualizing fine-tuning data...")
    plot_feature_distribution(finetune_data, "Node Feature Distribution (Fine-tuning Data)")
    plot_edge_distribution(finetune_data, "Edge Count Distribution (Fine-tuning Data)")
    plot_label_distribution(finetune_data, "Label Distribution (Fine-tuning Data)")

if __name__ == "__main__":
    dataset_path = "./processed_data"  # Modify to the actual processed_data path
    visualize_processed_data(dataset_path)