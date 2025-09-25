import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGNN(nn.Module):
    """
    简单的GNN实现，不依赖PyG库
    """
    def __init__(self, in_dim, out_dim):
        super(SimpleGNN, self).__init__()
        self.W = nn.Linear(in_dim, out_dim)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        简单的消息传递
        x: 节点特征 [num_nodes, in_dim]
        edge_index: 边索引 [2, num_edges]
        edge_weight: 边权重 [num_edges]
        """
        # 节点转换
        h = self.W(x)
        
        # 准备消息传递
        row, col = edge_index[0], edge_index[1]
        
        # 创建消息
        messages = h[col]
        
        # 聚合消息（直接求和）
        out = torch.zeros_like(h)
        for i in range(row.size(0)):
            src, dst = col[i], row[i]
            out[dst] += messages[i]
        
        # 应用非线性激活
        return F.relu(out)

class SimpleGCNLayer(nn.Module):
    """
    简化版GCN层，不依赖PyG库
    """
    def __init__(self, in_channels, out_channels):
        super(SimpleGCNLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        简化的GCN消息传递，优化版本
        """
        # 线性变换
        h = self.linear(x)
        
        # 获取源节点和目标节点
        row, col = edge_index[0], edge_index[1]
        
        # 边界检查和调整，不移除，而是剪裁超出范围的索引
        num_nodes = x.size(0)
        out_of_bounds_mask = (row >= num_nodes) | (col >= num_nodes)
        
        if out_of_bounds_mask.any():
            invalid_edges = out_of_bounds_mask.sum().item()
            print(f"SimpleGCNLayer: 检测到{invalid_edges}条超出范围的边索引，进行剪裁")
            
            # 剪裁索引而不是删除边
            row = torch.clamp(row, 0, num_nodes - 1)
            col = torch.clamp(col, 0, num_nodes - 1)
        
        # 快速计算度矩阵的逆平方根（用于归一化），使用bincount加速
        try:
            # 使用快速方法计算度
            deg = torch.bincount(row, minlength=num_nodes) + torch.bincount(col, minlength=num_nodes)
            # 避免除零错误
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # 使用矩阵方式计算消息传递（更高效）
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            # 聚合消息
            out = torch.zeros_like(h)
            src_normalized = norm.view(-1, 1) * h[col]
            # 使用index_add_加速聚合
            out.index_add_(0, row, src_normalized)
        except Exception as e:
            # 如果快速方法失败，回退到原始循环方式
            print(f"使用快速方法失败: {e}，回退到循环方式")
            
            # 1. 计算度矩阵的逆平方根（用于归一化）
            degree = torch.zeros(num_nodes, device=x.device)
            for i in range(row.size(0)):
                degree[row[i]] += 1
                degree[col[i]] += 1
            
            # 避免除零错误
            deg_inv_sqrt = degree.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            
            # 2. 消息传递 - 循环方式
            out = torch.zeros_like(h)
            for i in range(row.size(0)):
                src, dst = col[i], row[i]
                # 归一化权重
                norm = deg_inv_sqrt[dst] * deg_inv_sqrt[src]
                # 消息传递
                out[dst] += norm * h[src]
            
        return out

class DeepSimpleGNN(nn.Module):
    """
    多层简单GNN，替代DeepGCN
    """
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=3):
        super(DeepSimpleGNN, self).__init__()
        self.layers = nn.ModuleList()
        
        # 输入层
        self.layers.append(SimpleGCNLayer(in_dim, hidden_dim))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(SimpleGCNLayer(hidden_dim, hidden_dim))
        
        # 输出层
        self.layers.append(SimpleGCNLayer(hidden_dim, out_dim))
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        """
        # 边界检查和调整，不移除边，而是剪裁索引
        num_nodes = x.size(0)
        row, col = edge_index[0], edge_index[1]
        out_of_bounds_mask = (row >= num_nodes) | (col >= num_nodes)
        
        if out_of_bounds_mask.any():
            # 剪裁索引，不移除边
            invalid_edges = out_of_bounds_mask.sum().item()
            print(f"DeepSimpleGNN: 检测到{invalid_edges}条超出范围的边索引，进行剪裁")
            
            edge_index_adjusted = edge_index.clone()
            edge_index_adjusted[0] = torch.clamp(edge_index[0], 0, num_nodes - 1)
            edge_index_adjusted[1] = torch.clamp(edge_index[1], 0, num_nodes - 1)
            edge_index = edge_index_adjusted
        
        # 前向传播
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x, edge_index, edge_weight))
        
        # 最后一层不用激活函数
        return self.layers[-1](x, edge_index, edge_weight) 