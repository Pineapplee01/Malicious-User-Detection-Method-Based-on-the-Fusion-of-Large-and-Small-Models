import pytorch_lightning as pl
from torch import nn
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
import numpy as np
import torch.nn.functional as F
import os
import sys
from tqdm import tqdm

# 添加路径以导入原SeGA模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../SeGA-main/Code')))
# 导入简单GNN实现
from simple_gnn import SimpleGNN, SimpleGCNLayer, DeepSimpleGNN

class SeGA_LLM(pl.LightningModule):
    def __init__(self, in_dim, embed_dim, dropout, temp, pretext_task, pretrain, device, encoder='SimpleGCN', args=None):
        super(SeGA_LLM, self).__init__()
        self.device_name = device

        # 基础设置
        self.lr = args.lr
        self.wd = args.wd
        self.pretext_task = pretext_task
        self.pretrain = pretrain
        self.temp = temp
        self.batch_size = args.batch_size
        self.embed_dim = embed_dim
        self.encoder = encoder
        self.in_dim = in_dim
        
        # 节点特征设置
        self.list = args.lst
        self.list_num = args.list_num
        
        # 用户特征通道
        self.list_cat_num = 1  # 列表分类特征数
        self.list_numeric_num = 4  # 列表数值特征数
        self.list_tweet_channel = 768  # 列表推文特征维度
        self.list_des_channel = 768  # 列表描述特征维度
        
        # LLM增强特征通道
        self.llm_enhancement_channel = args.llm_enhancement_channel
        
        # 特征处理层 - 为每种特征类型创建专用线性层
        self.numeric_layer = nn.Linear(self.list_numeric_num, embed_dim)
        self.cat_layer = nn.Linear(self.list_cat_num, embed_dim)
        self.tweet_layer = nn.Linear(self.list_tweet_channel, embed_dim)
        self.des_layer = nn.Linear(self.list_des_channel, embed_dim)
        self.llm_layer = nn.Linear(self.llm_enhancement_channel, embed_dim)
        
        self.hidden_layer = nn.Linear(embed_dim * 5, embed_dim)  # 五个特征通道拼接

        # 图神经网络层
        if encoder == 'DeepGCN':
            try:
                # 尝试导入PyG库
                from torch_geometric.nn import DeepGCNLayer, GENConv
                print("使用PyG的DeepGCN实现")
                self.conv_layers = nn.ModuleList()
                for _ in range(3):  # 3层GCN
                    self.conv_layers.append(
                        DeepGCNLayer(
                            GENConv(embed_dim, embed_dim, aggr='softmax', learn_t=True, num_layers=2),
                            norm=nn.LayerNorm(embed_dim, elementwise_affine=True),
                            act=nn.ReLU(inplace=True)
                        )
                    )
            except ImportError:
                # 如果导入失败，使用简单实现
                print("PyG库导入失败，使用简单GCN实现")
                self.conv_layers = nn.ModuleList()
                for _ in range(3):
                    self.conv_layers.append(SimpleGCNLayer(embed_dim, embed_dim))
        else:
            # 使用自定义简单GNN实现
            print("使用自定义SimpleGCN实现")
            self.conv_layers = nn.ModuleList()
            for _ in range(3):
                self.conv_layers.append(SimpleGCNLayer(embed_dim, embed_dim))
        
        # 输出层
        self.out_layer = nn.Linear(embed_dim, embed_dim)
        self.prompt_mlp = nn.Linear(768, embed_dim)
        self.node_mlp = nn.Linear(embed_dim, embed_dim)
        
        # 分类器 - 为恶意账号检测设置3类输出 (人类、机器人、恶意用户)
        self.classifier = nn.Linear(embed_dim, 3)

        # 激活、损失函数等
        self.dropout = nn.Dropout(dropout)
        self.ce_loss = nn.CrossEntropyLoss()
        self.relu = nn.ReLU()
        
        # 初始化权重
        self.init_weight()
        
        # 预训练设置
        if pretrain == True:
            # 修改为相对路径
            prompt_embedding_path = os.path.join(args.dataset_path, "prompt_embeddings/")
            if not os.path.exists(prompt_embedding_path):
                
                # 如果路径不存在，创建目录并生成随机嵌入
                os.makedirs(prompt_embedding_path, exist_ok=True)
                print(f"创建随机预训练嵌入矩阵 ({args.template})")
                
                # 创建随机嵌入矩阵 (100, 768)
                random_embeddings = torch.randn(100, 768)
                torch.save(random_embeddings, os.path.join(prompt_embedding_path, f"{args.prompt_encoder}_{args.template}.pt"))
                
                # 创建随机非零字典
                none_zero = np.arange(100)
                torch.save(none_zero, os.path.join(prompt_embedding_path, "none_zero.pt"))
                if args.template not in ["l", "s", "n"]:
                    torch.save(none_zero, os.path.join(prompt_embedding_path, f"none_zero_{args.template}.pt"))
            
            try:
                self.pretrain_embedding_dict = torch.load(os.path.join(prompt_embedding_path, f"{args.prompt_encoder}_{args.template}.pt"), map_location=device)
                if args.template in ["l", "s", "n"]:
                    self.none_zero_dict = np.array(torch.load(os.path.join(prompt_embedding_path, "none_zero.pt"), map_location=device))
                else:
                    self.none_zero_dict = np.array(torch.load(os.path.join(prompt_embedding_path, f"none_zero_{args.template}.pt"), map_location=device))
           
            except Exception as e:
                print(f"加载预训练嵌入失败: {e}，将创建随机嵌入")
                # 创建随机嵌入矩阵和非零字典
                self.pretrain_embedding_dict = torch.randn(100, 768)
                self.none_zero_dict = np.arange(100)
                
        if pretext_task == "multi":
            self.multi_classifier = nn.Linear(embed_dim, 153)
            self.bce_loss = nn.BCEWithLogitsLoss()

        # 创建日志字典，避免使用self.log
        self.log_dict = {}

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
    
    def fix_tensor_type(self, tensor):
        """
        修复张量类型，确保所有数值张量都是 float32 类型
        """
        if tensor is None:
            return None
            
        # 检查张量是否为空
        if tensor.numel() == 0:
            return tensor
            
        # 如果是数值张量且不是float32，则转换
        if tensor.dtype in [torch.float16, torch.float64, torch.int32, torch.int64]:
            return tensor.to(torch.float32)
        return tensor
        
    def training_step(self, train_batch, batch_idx):
        """
        训练步骤实现
        """
        # 检查输入维度
        if train_batch.x is None or train_batch.x.shape[0] == 0:
            raise ValueError("输入张量维度为空或无效")

        # 检查设备
        self.device_name = train_batch.x.device
        
        # 节点特征编码
        features = torch.zeros(train_batch.num_nodes, self.embed_dim, dtype=torch.float32, device=self.device_name)
        
        # 用户节点特征
        user_mask = (train_batch.n_id >= 0) & (train_batch.n_id < 100001)
        if user_mask.sum() > 0:  # 只有在存在用户节点时才处理
            user_x = train_batch.x.clone()
            user_x = self.fix_tensor_type(user_x[user_mask])
            
            # 检查并打印各特征维度 - 仅在首次运行时
            if batch_idx == 0 and not hasattr(self, "_dim_reported"):
                print(f"用户特征原始维度: {user_x.shape}")
                self._dim_reported = True
            
            # 动态确定特征维度
            feature_dims = user_x.shape[1]
            llm_start = self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel
            llm_dim = feature_dims - llm_start
            
            # 如果LLM维度不匹配，需要调整线性层 - 仅在首次检测时
            if llm_dim != self.llm_enhancement_channel and not hasattr(self, "_dim_adjusted"):
                print(f"检测到LLM特征维度: {llm_dim}，预期维度: {self.llm_enhancement_channel}")
                print(f"重新创建LLM线性层以适应维度: {llm_dim}")
                self.llm_enhancement_channel = llm_dim
                self.llm_layer = nn.Linear(llm_dim, self.embed_dim).to(self.device_name)
                self._dim_adjusted = True
            
            # 检查特征划分是否有效
            if user_x.shape[1] < llm_start:
                raise ValueError(f"用户特征维度不足: {user_x.shape[1]}, 需要至少 {llm_start}")
                
            user_cat_features = user_x[:, :self.list_cat_num]
            user_prop_features = user_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
            user_tweet_features = user_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
            user_des_features = user_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: llm_start]
            
            # 确保LLM特征具有正确的维度
            user_llm_features = user_x[:, llm_start:]
            
            # 仅在首次运行时打印特征维度
            if batch_idx == 0 and hasattr(self, "_dim_reported") and not hasattr(self, "_feature_dims_reported"):
                print(f"特征维度检查: cat={user_cat_features.shape}, prop={user_prop_features.shape}, "
                     f"tweet={user_tweet_features.shape}, des={user_des_features.shape}, llm={user_llm_features.shape}")
                self._feature_dims_reported = True
            
            # 用户特征处理 - 使用不同的线性层
            user_features_numeric = self.dropout(self.relu(self.numeric_layer(user_prop_features)))
            user_features_bool = self.dropout(self.relu(self.cat_layer(user_cat_features)))
            user_features_tweet = self.dropout(self.relu(self.tweet_layer(user_tweet_features)))
            user_features_des = self.dropout(self.relu(self.des_layer(user_des_features)))
            user_features_llm = self.dropout(self.relu(self.llm_layer(user_llm_features)))
            
            user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, 
                                     user_features_des, user_features_llm), dim=1)
            user_features = self.dropout(self.relu(self.hidden_layer(user_features)))
            features[user_mask] = user_features

        # 处理列表节点（如果有）
        if self.list:
            list_mask = (train_batch.n_id >= 100001) & (train_batch.n_id < 100001 + self.list_num)
            if list_mask.sum() > 0:  # 只有在存在列表节点时才处理
                list_x = train_batch.x.clone()
                list_x = self.fix_tensor_type(list_x[list_mask])
                
                # 确保LLM特征具有正确的维度
                llm_start = self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel
                
                # 检查特征划分是否有效
                if list_x.shape[1] < llm_start:
                    raise ValueError(f"列表特征维度不足: {list_x.shape[1]}, 需要至少 {llm_start}")
                
                list_cat_features = list_x[:, :self.list_cat_num]
                list_prop_features = list_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                list_tweet_features = list_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                list_des_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: llm_start]
                list_llm_features = list_x[:, llm_start:]

                # 特征处理
                list_features_numeric = self.dropout(self.relu(self.numeric_layer(list_prop_features)))
                list_features_bool = self.dropout(self.relu(self.cat_layer(list_cat_features)))
                list_features_tweet = self.dropout(self.relu(self.tweet_layer(list_tweet_features)))
                list_features_des = self.dropout(self.relu(self.des_layer(list_des_features)))
                list_features_llm = self.dropout(self.relu(self.llm_layer(list_llm_features)))

                # 拼接特征
                list_features = torch.cat((list_features_numeric, list_features_bool, list_features_tweet, 
                                        list_features_des, list_features_llm), dim=1)
                list_features = self.dropout(self.relu(self.hidden_layer(list_features)))
                features[list_mask] = list_features

        ### 图神经网络处理 ###
        edge_index = train_batch.edge_index
        
        # 剪裁边索引的优化版本 - 仅处理一次
        if not hasattr(self, "_edge_index_clipped") and edge_index is not None:
            self._edge_index_clipped = True
            num_nodes = train_batch.num_nodes
            row, col = edge_index[0], edge_index[1]
            out_of_bounds_mask = (row >= num_nodes) | (col >= num_nodes)
            
            if out_of_bounds_mask.any():
                invalid_edges = out_of_bounds_mask.sum().item()
                print(f"训练阶段：将剪裁{invalid_edges}条超出范围的边索引（仅执行一次）")
                
                # 创建调整后的边索引张量
                train_batch.edge_index[0] = torch.clamp(edge_index[0], 0, num_nodes - 1)
                train_batch.edge_index[1] = torch.clamp(edge_index[1], 0, num_nodes - 1)
                
                print("边索引剪裁完成，继续训练...")
            else:
                print("边索引检查完成，没有超出范围的边索引")
        
        # 获取边索引和类型
        edge_index = train_batch.edge_index
        
        # 提取edge_type (如果有)
        if hasattr(train_batch, 'edge_attr') and train_batch.edge_attr is not None:
            edge_type = train_batch.edge_attr.view(-1)
        else:
            edge_type = None
            
        # 应用GNN层
        try:
            # 尝试使用PyG风格接口
            features = self.relu(self.conv_layers[0](features, edge_index, edge_type))
            features = self.relu(self.conv_layers[1](features, edge_index, edge_type))
            features = self.relu(self.conv_layers[2](features, edge_index, edge_type))
        except Exception as e:
            print(f"使用PyG风格GNN接口失败: {e}，尝试使用简单GNN...")
            # 使用简单GNN接口
            for layer in self.conv_layers:
                features = self.relu(layer(features, edge_index))
        
        if self.pretrain == True:
            # 预训练任务
            if self.pretext_task == "contrastive":
                ### 自对比学习 ###
                features = self.dropout(self.relu(self.out_layer(features)))
                anchor = self.node_mlp(features)
                anchor = anchor[user_mask.nonzero(as_tuple=True)[0]]

                # 对比学习损失计算
                prompt_embedding = self.prompt_mlp(self.pretrain_embedding_dict)
                prompt_embedding = prompt_embedding[self.none_zero_dict]
                
                # 计算点积
                sim_matrix = torch.matmul(anchor, prompt_embedding.T) / self.temp
                
                # 计算对比损失
                logits = torch.cat([sim_matrix], dim=0)
                label = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

                loss = self.ce_loss(logits.float(), label)
                self.custom_log("pre_train_ce", loss, prog_bar=True)
                return loss

            elif self.pretext_task == "multi":
                features = self.dropout(self.relu(self.out_layer(features)))
                pred = self.multi_classifier(features)
                pred = pred[user_mask.nonzero(as_tuple=True)[0]]

                loss = self.bce_loss(pred.float(), label.float())
                return loss
        else:
            # 微调/训练阶段
            mask = train_batch.train_mask
            
            # 检查掩码是否存在
            if mask is None or mask.sum() == 0:
                print("警告: 训练掩码为空或不存在")
                # 创建一个默认掩码
                mask = torch.zeros(train_batch.num_nodes, dtype=torch.bool, device=features.device)
                # 使用用户节点作为默认掩码
                mask[user_mask] = True
            
            # 检查标签是否存在
            if not hasattr(train_batch, 'y') or train_batch.y is None:
                print("警告: 标签不存在")
                return torch.tensor(0.0, requires_grad=True, device=features.device)
            
            # 检查标签与掩码大小是否匹配
            if train_batch.y.size(0) != mask.size(0):
                print(f"警告: 标签大小 ({train_batch.y.size(0)}) 与掩码大小 ({mask.size(0)}) 不匹配")
                
                # 如果标签大小与user_mask匹配，直接使用user_mask
                if train_batch.y.size(0) == user_mask.sum().item():
                    print("使用用户掩码作为训练掩码")
                    # 提取用户节点特征
                    features_masked = features[user_mask]
                    label = train_batch.y.long()
                else:
                    print("使用掩码筛选标签")
                    # 确保label大小与mask匹配
                    if mask.sum().item() > train_batch.y.size(0):
                        # 裁剪mask以匹配label大小
                        mask_indices = mask.nonzero(as_tuple=True)[0]
                        mask = torch.zeros_like(mask)
                        mask[mask_indices[:train_batch.y.size(0)]] = True
                    
                    # 提取mask位置的特征
                    features_masked = features[mask]
                    label = train_batch.y.long()
            else:
                # 正常情况：标签与掩码大小匹配
                features_masked = features[mask]
                label = train_batch.y.long()[mask]

            # 输出调试信息
            if batch_idx == 0:
                print(f"标签形状: {label.shape}, 特征形状: {features_masked.shape}")

            ### 分类 ###
            features_masked = self.dropout(self.relu(self.out_layer(features_masked)))
            pred = self.classifier(features_masked)
            
            # 输出调试信息
            if batch_idx == 0:
                print(f"预测形状: {pred.shape}, 标签形状: {label.shape}")
                print(f"预测值范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print(f"标签范围: [{label.min().item()}, {label.max().item()}]")

            # 计算性能指标
            try:
                topk_preds = pred.topk(1, dim=1)[1]
                topk_preds = topk_preds.squeeze(1)
                
                if batch_idx == 0:
                    print(f"topk_preds形状: {topk_preds.shape}, 标签形状: {label.shape}")
                
                # 确保大小匹配
                if topk_preds.size(0) == label.size(0):
                    acc = (topk_preds == label).float().mean()
                    
                    # 计算F1分数
                    target = label.cpu().detach().numpy()
                    predict = topk_preds.cpu().detach().numpy()
                    f1 = f1_score(target, predict, average='macro')
                    
                    # 记录指标
                    self.custom_log("train_acc", acc, prog_bar=True)
                    self.custom_log("train_f1", f1, prog_bar=True)
                else:
                    print(f"警告: topk_preds形状 ({topk_preds.size(0)}) 与标签形状 ({label.size(0)}) 不匹配")
                    acc = torch.tensor(0.0, device=features.device)
                    f1 = 0.0
            except Exception as e:
                print(f"计算指标时出错: {e}")
                acc = torch.tensor(0.0, device=features.device)
                f1 = 0.0

            # 计算损失
            try:
                loss = self.ce_loss(pred.float(), label.long())
                if batch_idx == 0:
                    print(f"损失值: {loss.item():.4f}")
                return loss
            except Exception as e:
                print(f"计算损失时出错: {e}")
                return torch.tensor(1.0, requires_grad=True, device=features.device)

    def validation_step(self, valid_batch, batch_idx):
        """
        验证步骤 - 恢复原有功能
        """
        with torch.no_grad():
            if not hasattr(valid_batch, 'val_mask') or valid_batch.val_mask is None:
                valid_batch.val_mask = torch.ones(valid_batch.num_nodes, dtype=torch.bool, device=self.device_name)
        
            # 节点特征编码，与训练步骤相同
            features = torch.zeros(valid_batch.num_nodes, self.embed_dim, dtype=torch.float32, device=self.device_name)
            
            # 用户节点特征
            user_mask = (valid_batch.n_id >= 0) & (valid_batch.n_id < 100001)
            if user_mask.sum() > 0:  # 只有在存在用户节点时才处理
                user_x = valid_batch.x.clone()
                user_x = self.fix_tensor_type(user_x[user_mask])
                
                # 确保LLM特征具有正确的维度
                llm_start = self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel
                
                # 检查特征划分是否有效
                if user_x.shape[1] < llm_start:
                    raise ValueError(f"用户特征维度不足: {user_x.shape[1]}, 需要至少 {llm_start}")
                
                user_cat_features = user_x[:, :self.list_cat_num]
                user_prop_features = user_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                user_tweet_features = user_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                user_des_features = user_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: llm_start]
                user_llm_features = user_x[:, llm_start:]
                
                # 用户特征处理
                user_features_numeric = self.dropout(self.relu(self.numeric_layer(user_prop_features)))
                user_features_bool = self.dropout(self.relu(self.cat_layer(user_cat_features)))
                user_features_tweet = self.dropout(self.relu(self.tweet_layer(user_tweet_features)))
                user_features_des = self.dropout(self.relu(self.des_layer(user_des_features)))
                user_features_llm = self.dropout(self.relu(self.llm_layer(user_llm_features)))
                
                user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, 
                                         user_features_des, user_features_llm), dim=1)
                user_features = self.dropout(self.relu(self.hidden_layer(user_features)))
                features[user_mask] = user_features
            
            # 处理列表节点（如果有）
            if self.list:
                list_mask = (valid_batch.n_id >= 100001) & (valid_batch.n_id < 100001 + self.list_num)
                if list_mask.sum() > 0:  # 只有在存在列表节点时才处理
                    list_x = valid_batch.x.clone()
                    list_x = self.fix_tensor_type(list_x[list_mask])
                    
                    # 检查特征划分是否有效
                    if list_x.shape[1] < self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel:
                        raise ValueError(f"列表特征维度不足: {list_x.shape[1]}, 需要至少 {self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel}")
                    
                    list_cat_features = list_x[:, :self.list_cat_num]
                    list_prop_features = list_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                    list_tweet_features = list_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                    list_des_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel]
                    list_llm_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel:]
                    
                    list_features_numeric = self.dropout(self.relu(self.numeric_layer(list_prop_features)))
                    list_features_bool = self.dropout(self.relu(self.cat_layer(list_cat_features)))
                    list_features_tweet = self.dropout(self.relu(self.tweet_layer(list_tweet_features)))
                    list_features_des = self.dropout(self.relu(self.des_layer(list_des_features)))
                    list_features_llm = self.dropout(self.relu(self.llm_layer(list_llm_features)))
                    
                    list_features = torch.cat((list_features_numeric, list_features_bool, list_features_tweet, 
                                             list_features_des, list_features_llm), dim=1)
                    list_features = self.dropout(self.relu(self.hidden_layer(list_features)))
                    features[list_mask] = list_features
            
            # 图神经网络处理
            edge_index = valid_batch.edge_index
            
            # 剪裁边索引的优化版本 - 仅处理一次
            if not hasattr(self, "_val_edge_index_clipped") and edge_index is not None:
                self._val_edge_index_clipped = True
                num_nodes = valid_batch.num_nodes
                row, col = edge_index[0], edge_index[1]
                out_of_bounds_mask = (row >= num_nodes) | (col >= num_nodes)
                
                if out_of_bounds_mask.any():
                    invalid_edges = out_of_bounds_mask.sum().item()
                    print(f"验证阶段：将剪裁{invalid_edges}条超出范围的边索引（仅执行一次）")
                    
                    # 创建调整后的边索引张量
                    valid_batch.edge_index[0] = torch.clamp(edge_index[0], 0, num_nodes - 1)
                    valid_batch.edge_index[1] = torch.clamp(edge_index[1], 0, num_nodes - 1)
                    
                    print("验证边索引剪裁完成")
            
            # 使用剪裁后的边索引
            edge_index = valid_batch.edge_index
            
            # 提取edge_type (如果有)
            if hasattr(valid_batch, 'edge_attr') and valid_batch.edge_attr is not None:
                edge_type = valid_batch.edge_attr.view(-1)
            else:
                edge_type = None
                
            # 应用GNN层
            try:
                # 尝试使用PyG风格接口
                features = self.relu(self.conv_layers[0](features, edge_index, edge_type))
                features = self.relu(self.conv_layers[1](features, edge_index, edge_type))
                features = self.relu(self.conv_layers[2](features, edge_index, edge_type))
            except Exception as e:
                print(f"使用PyG风格GNN接口失败: {e}，尝试使用简单GNN...")
                # 使用简单GNN接口
                for layer in self.conv_layers:
                    features = self.relu(layer(features, edge_index))
            
            # 输出层处理
            features = self.dropout(self.relu(self.out_layer(features)))
            pred = self.classifier(features)
            
            # 验证集指标计算
            mask = valid_batch.val_mask
            label = valid_batch.y.long()[mask]
            
            pred = pred[mask.nonzero(as_tuple=True)[0]]
            
            topk_preds = pred.topk(1, dim=1)[1].squeeze(1)
            val_acc = (topk_preds == label).float().mean()
            
            target = label.cpu().detach().numpy()
            predict = topk_preds.cpu().detach().numpy()
            f1 = f1_score(target, predict, average='macro')
            
            val_loss = self.ce_loss(pred.float(), label.long())
            
            # 记录验证指标
            self.custom_log("val_loss", val_loss, prog_bar=True)
            self.custom_log("val_acc", val_acc, prog_bar=True)
            self.custom_log("val_f1", f1, prog_bar=True)
            
            return {"val_loss": val_loss, "val_acc": val_acc}

    def test_step(self, test_batch, batch_idx):
        """
        测试步骤 - 恢复原有功能
        """
        with torch.no_grad():
            if not hasattr(test_batch, 'test_mask') or test_batch.test_mask is None:
                test_batch.test_mask = torch.ones(test_batch.num_nodes, dtype=torch.bool, device=self.device_name)
            
            # 节点特征编码，与训练和验证步骤相同
            features = torch.zeros(test_batch.num_nodes, self.embed_dim, dtype=torch.float32, device=self.device_name)
            
            # 处理用户节点
            user_mask = (test_batch.n_id >= 0) & (test_batch.n_id < 100001)
            if user_mask.sum() > 0:  # 只有在存在用户节点时才处理
                user_x = test_batch.x.clone()
                user_x = self.fix_tensor_type(user_x[user_mask])
                
                # 确保LLM特征具有正确的维度
                llm_start = self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel
                
                # 检查特征划分是否有效
                if user_x.shape[1] < llm_start:
                    raise ValueError(f"用户特征维度不足: {user_x.shape[1]}, 需要至少 {llm_start}")
                
                user_cat_features = user_x[:, :self.list_cat_num]
                user_prop_features = user_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                user_tweet_features = user_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                user_des_features = user_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: llm_start]
                user_llm_features = user_x[:, llm_start:]
                
                user_features_numeric = self.dropout(self.relu(self.numeric_layer(user_prop_features)))
                user_features_bool = self.dropout(self.relu(self.cat_layer(user_cat_features)))
                user_features_tweet = self.dropout(self.relu(self.tweet_layer(user_tweet_features)))
                user_features_des = self.dropout(self.relu(self.des_layer(user_des_features)))
                user_features_llm = self.dropout(self.relu(self.llm_layer(user_llm_features)))
                
                user_features = torch.cat((user_features_numeric, user_features_bool, user_features_tweet, 
                                         user_features_des, user_features_llm), dim=1)
                user_features = self.dropout(self.relu(self.hidden_layer(user_features)))
                features[user_mask] = user_features
            
            # 处理列表节点（如果有）
            if self.list:
                list_mask = (test_batch.n_id >= 100001) & (test_batch.n_id < 100001 + self.list_num)
                if list_mask.sum() > 0:  # 只有在存在列表节点时才处理
                    list_x = test_batch.x.clone()
                    list_x = self.fix_tensor_type(list_x[list_mask])
                    
                    # 检查特征划分是否有效
                    if list_x.shape[1] < self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel:
                        raise ValueError(f"列表特征维度不足: {list_x.shape[1]}, 需要至少 {self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel}")
                    
                    list_cat_features = list_x[:, :self.list_cat_num]
                    list_prop_features = list_x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
                    list_tweet_features = list_x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
                    list_des_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel]
                    list_llm_features = list_x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel:]
                    
                    list_features_numeric = self.dropout(self.relu(self.numeric_layer(list_prop_features)))
                    list_features_bool = self.dropout(self.relu(self.cat_layer(list_cat_features)))
                    list_features_tweet = self.dropout(self.relu(self.tweet_layer(list_tweet_features)))
                    list_features_des = self.dropout(self.relu(self.des_layer(list_des_features)))
                    list_features_llm = self.dropout(self.relu(self.llm_layer(list_llm_features)))
                    
                    list_features = torch.cat((list_features_numeric, list_features_bool, list_features_tweet, 
                                             list_features_des, list_features_llm), dim=1)
                    list_features = self.dropout(self.relu(self.hidden_layer(list_features)))
                    features[list_mask] = list_features
            
            # 图神经网络处理
            edge_index = test_batch.edge_index
            
            # 剪裁边索引的优化版本 - 仅处理一次
            if not hasattr(self, "_test_edge_index_clipped") and edge_index is not None:
                self._test_edge_index_clipped = True
                num_nodes = test_batch.num_nodes
                row, col = edge_index[0], edge_index[1]
                out_of_bounds_mask = (row >= num_nodes) | (col >= num_nodes)
                
                if out_of_bounds_mask.any():
                    invalid_edges = out_of_bounds_mask.sum().item()
                    print(f"测试阶段：将剪裁{invalid_edges}条超出范围的边索引（仅执行一次）")
                    
                    # 创建调整后的边索引张量
                    test_batch.edge_index[0] = torch.clamp(edge_index[0], 0, num_nodes - 1)
                    test_batch.edge_index[1] = torch.clamp(edge_index[1], 0, num_nodes - 1)
                    
                    print("测试边索引剪裁完成")
            
            # 使用剪裁后的边索引
            edge_index = test_batch.edge_index
            
            # 提取edge_type (如果有)
            if hasattr(test_batch, 'edge_attr') and test_batch.edge_attr is not None:
                edge_type = test_batch.edge_attr.view(-1)
            else:
                edge_type = None
                
            # 应用GNN层
            try:
                # 尝试使用PyG风格接口
                features = self.relu(self.conv_layers[0](features, edge_index, edge_type))
                features = self.relu(self.conv_layers[1](features, edge_index, edge_type))
                features = self.relu(self.conv_layers[2](features, edge_index, edge_type))
            except Exception as e:
                print(f"使用PyG风格GNN接口失败: {e}，尝试使用简单GNN...")
                # 使用简单GNN接口
                for layer in self.conv_layers:
                    features = self.relu(layer(features, edge_index))
            
            # 输出层处理
            features = self.dropout(self.relu(self.out_layer(features)))
            pred = self.classifier(features)
            
            # 测试集指标计算
            mask = test_batch.test_mask
            label = test_batch.y.long()[mask]
            
            pred = pred[mask.nonzero(as_tuple=True)[0]]
            
            topk_preds = pred.topk(1, dim=1)[1].squeeze(1)
            test_acc = (topk_preds == label).float().mean()
            
            target = label.cpu().detach().numpy()
            predict = topk_preds.cpu().detach().numpy()
            f1 = f1_score(target, predict, average='macro')
            
            test_loss = self.ce_loss(pred.float(), label.long())
            
            # 记录测试指标
            self.custom_log("test_loss", test_loss, prog_bar=True)
            self.custom_log("test_acc", test_acc, prog_bar=True)
            self.custom_log("test_f1", f1, prog_bar=True)
            
            return {"test_loss": test_loss, "test_acc": test_acc, "test_f1": f1}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def pretrain(self, data, epoch, batch_size, writer, device, embedding_path=None, use_tqdm=False):
        """
        执行预训练过程，使用简单批处理代替NeighborLoader
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        best_loss = float('inf')
        
        # 检查索引是否有效
        if not hasattr(data, 'pretrain_train_idx') or data.pretrain_train_idx is None or len(data.pretrain_train_idx) == 0:
            raise ValueError("预训练索引无效或为空")
            
        # 替代NeighborLoader的简单批处理
        def create_batches(indices, batch_size):
            return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        
        train_batches = create_batches(data.pretrain_train_idx, batch_size)
        
        # 将数据移动到设备
        data = data.to(device)
        
        # 主训练循环
        epoch_progress = None
        if use_tqdm:
            epoch_progress = tqdm(range(epoch), desc="预训练进度", unit="epoch", position=0)
        
        for e in range(epoch):
            epoch_loss = 0.0
            self.train()
            num_batches = 0
            
            # 更新总体进度
            if epoch_progress is not None:
                epoch_progress.update(1)
                epoch_progress.set_description(f"预训练 Epoch {e+1}/{epoch}")
            
            # 创建批次进度条 - 使用position=1放在总进度条下方
            batch_iter = train_batches
            if use_tqdm:
                batch_iter = tqdm(train_batches, 
                                 desc=f"处理批次", 
                                 leave=False,       # 完成后不保留进度条
                                 unit="batch", 
                                 position=1,       # 放在第二行
                                 ncols=100,        # 限制显示宽度
                                 disable=False,    # 不禁用
                                 mininterval=1.0)  # 最小更新间隔秒数
                # 保存当前进度条的引用
                self.current_progress_bar = batch_iter
            
            # 处理每个batch
            for i, batch_indices in enumerate(batch_iter):
                # 创建训练掩码
                train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
                train_mask[batch_indices] = True
                data.train_mask = train_mask
                
                # 将n_id设置为全部节点的索引
                data.n_id = torch.arange(data.num_nodes, device=device)
                
                # 前向传播和损失计算
                optimizer.zero_grad()
                
                try:
                    loss = self.training_step(data, i)
                    
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                    
                    # 更新进度条
                    if use_tqdm and i % 10 == 0:  # 减少更新频率
                        batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})
                        
                except Exception as e:
                    print(f"批次{i}训练出错: {e}")
                
                # 避免内存泄漏
                torch.cuda.empty_cache()
            
            # 计算平均损失
            avg_loss = epoch_loss / max(1, num_batches)
            
            # 记录到tensorboard
            writer.add_scalar("pretrain/loss", avg_loss, e)
            
            # 保存最佳模型
            if avg_loss < best_loss:
                best_loss = avg_loss
                if embedding_path:
                    state_dict_path = f"{embedding_path}_best_model.pt"
                    torch.save(self.state_dict(), state_dict_path)
                    print(f"保存最佳模型到 {state_dict_path}, 损失: {best_loss:.4f}")
            
            if (e + 1) % 10 == 0 or e == 0 or e == epoch-1:
                print(f"预训练 Epoch {e+1}/{epoch}, Loss: {avg_loss:.4f}")
        
        # 关闭总进度条
        if epoch_progress is not None:
            epoch_progress.close()
        
        # 保存嵌入
        if embedding_path:
            with torch.no_grad():
                self.eval()
                # 提取节点嵌入
                print("计算并保存最终嵌入...")
                embeddings = self.get_embeddings(data)
                torch.save(embeddings, f"{embedding_path}_embeddings.pt")
        
        return {"loss": best_loss}

    def finetune(self, data, epoch, batch_size, writer, device, embedding_path=None, use_tqdm=False):
        """
        执行微调过程，使用简单批处理代替NeighborLoader
        """
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, verbose=True)
        best_val_acc = 0.0
        best_val_f1 = 0.0
        
        # 检查索引是否有效
        if not hasattr(data, 'finetune_train_idx') or data.finetune_train_idx is None or len(data.finetune_train_idx) == 0:
            raise ValueError("微调训练索引无效或为空")
        if not hasattr(data, 'finetune_valid_idx') or data.finetune_valid_idx is None or len(data.finetune_valid_idx) == 0:
            raise ValueError("微调验证索引无效或为空")
            
        # 替代NeighborLoader的简单批处理
        def create_batches(indices, batch_size):
            return [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
        
        train_batches = create_batches(data.finetune_train_idx, batch_size)
        val_batches = create_batches(data.finetune_valid_idx, batch_size)
        
        # 将数据移动到设备
        data = data.to(device)
        
        # 创建总进度条
        epoch_progress = None
        if use_tqdm:
            epoch_progress = tqdm(range(epoch), desc="微调进度", unit="epoch", position=0)
        
        # 主训练循环
        for e in range(epoch):
            # 更新总体进度
            if epoch_progress is not None:
                epoch_progress.update(1)
                epoch_progress.set_description(f"微调 Epoch {e+1}/{epoch}")
            
            # 训练阶段
            epoch_loss = 0.0
            train_acc = 0.0
            train_f1 = 0.0
            self.train()
            
            # 创建批次进度条
            batch_iter = train_batches
            if use_tqdm:
                batch_iter = tqdm(train_batches, 
                                 desc=f"训练批次", 
                                 leave=False,       # 完成后不保留进度条
                                 unit="batch", 
                                 position=1,       # 放在第二行
                                 ncols=100,        # 限制显示宽度 
                                 mininterval=1.0)  # 最小更新间隔秒数
                # 保存当前进度条的引用
                self.current_progress_bar = batch_iter
            
            # 处理每个训练batch
            for i, batch_indices in enumerate(batch_iter):
                # 创建训练掩码
                train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
                train_mask[batch_indices] = True
                data.train_mask = train_mask
                
                # 前向传播和损失计算
                optimizer.zero_grad()
                
                # 将n_id设置为全部节点的索引
                data.n_id = torch.arange(data.num_nodes, device=device)
                
                loss = self.training_step(data, i)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 更新进度条
                if use_tqdm and i % 10 == 0:  # 减少更新频率
                    batch_iter.set_postfix({"loss": f"{loss.item():.4f}"})
                
                # 避免内存泄漏
                torch.cuda.empty_cache()
            
            # 计算平均训练损失
            avg_train_loss = epoch_loss / max(1, len(train_batches))
            
            # 验证阶段
            self.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_f1 = 0.0
            val_count = 0
            
            # 创建验证批次进度条
            val_iter = val_batches
            if use_tqdm:
                val_iter = tqdm(val_batches, 
                               desc=f"验证批次", 
                               leave=False,      # 完成后不保留进度条
                               unit="batch", 
                               position=1,       # 放在第二行
                               ncols=100,        # 限制显示宽度
                               mininterval=1.0)  # 最小更新间隔秒数
                # 保存当前进度条的引用
                self.current_progress_bar = val_iter
            
            with torch.no_grad():
                for i, batch_indices in enumerate(val_iter):
                    # 创建验证掩码
                    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
                    val_mask[batch_indices] = True
                    data.val_mask = val_mask
                    
                    # 将n_id设置为全部节点的索引
                    data.n_id = torch.arange(data.num_nodes, device=device)
                    
                    # 验证步骤
                    result = self.validation_step(data, i)
                    val_loss += result["val_loss"].item()
                    val_acc += result["val_acc"].item()
                    val_count += 1
                    
                    # 更新进度条
                    if use_tqdm and i % 10 == 0:  # 减少更新频率
                        val_iter.set_postfix({
                            "loss": f"{result['val_loss'].item():.4f}", 
                            "acc": f"{result['val_acc'].item():.4f}"
                        })
                    
                    # 避免内存泄漏
                    torch.cuda.empty_cache()
            
            # 计算平均验证指标
            avg_val_loss = val_loss / max(1, val_count)
            avg_val_acc = val_acc / max(1, val_count)
            
            # 学习率调度器更新
            scheduler.step(avg_val_loss)
            
            # 记录到tensorboard
            writer.add_scalar("finetune/train_loss", avg_train_loss, e)
            writer.add_scalar("finetune/val_loss", avg_val_loss, e)
            writer.add_scalar("finetune/val_acc", avg_val_acc, e)
            
            # 清除当前进度条引用
            self.current_progress_bar = None
            
            # 保存最佳模型
            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                if embedding_path:
                    state_dict_path = f"{embedding_path}_best_model.pt"
                    torch.save(self.state_dict(), state_dict_path)
                    print(f"保存最佳模型到 {state_dict_path}, 准确率: {best_val_acc:.4f}")
            
            # 每个epoch结束打印统计信息 - 更新到主进度条
            if epoch_progress is not None:
                epoch_progress.set_postfix({
                    "训练损失": f"{avg_train_loss:.4f}", 
                    "验证损失": f"{avg_val_loss:.4f}", 
                    "验证准确率": f"{avg_val_acc:.4f}"
                })
            else:
                print(f"微调 Epoch {e+1}/{epoch}, 训练损失: {avg_train_loss:.4f}, 验证损失: {avg_val_loss:.4f}, 验证准确率: {avg_val_acc:.4f}")
        
        # 关闭总进度条
        if epoch_progress is not None:
            epoch_progress.close()
        
        # 保存嵌入
        if embedding_path:
            with torch.no_grad():
                self.eval()
                # 提取节点嵌入
                print("计算并保存最终嵌入...")
                embeddings = self.get_embeddings(data)
                torch.save(embeddings, f"{embedding_path}_embeddings.pt")
        
        return {"val_acc": best_val_acc}
    
    def get_embeddings(self, data):
        """
        计算节点嵌入
        """
        self.eval()
        with torch.no_grad():
            # 创建掩码来包含所有节点
            all_nodes_mask = torch.ones(data.num_nodes, dtype=torch.bool, device=data.x.device)
            
            # 初始化特征空间
            features = torch.zeros(data.num_nodes, self.embed_dim, dtype=torch.float32, device=data.x.device)
            
            # 提取用户节点特征
            user_mask = (data.n_id >= 0) & (data.n_id < 100001)
            if user_mask.any() and user_mask.sum() > 0:
                user_x = data.x[user_mask]
                
                # 检查特征维度
                if user_x.shape[1] < self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel:
                    raise ValueError(f"用户特征维度不足: {user_x.shape[1]}, 需要至少 {self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel}")
                
                # 特征提取逻辑
                user_features = self.extract_node_features(user_x)
                features[user_mask] = user_features
            
            # 处理列表节点（如果有）
            if self.list:
                list_mask = (data.n_id >= 100001) & (data.n_id < 100001 + self.list_num)
                if list_mask.any() and list_mask.sum() > 0:
                    list_x = data.x[list_mask]
                    
                    # 检查特征维度
                    if list_x.shape[1] < self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel:
                        raise ValueError(f"列表特征维度不足: {list_x.shape[1]}, 需要至少 {self.list_cat_num + self.list_numeric_num + self.list_tweet_channel + self.list_des_channel}")
                    
                    # 特征提取逻辑
                    list_features = self.extract_node_features(list_x)
                    features[list_mask] = list_features
            
            # 应用图神经网络
            if hasattr(data, 'edge_index') and data.edge_index is not None and data.edge_index.size(1) > 0:
                edge_index = data.edge_index
                
                # 检查边类型
                if hasattr(data, 'edge_attr') and data.edge_attr is not None and data.edge_attr.numel() > 0:
                    edge_type = data.edge_attr.view(-1)
                    
                    # 检查边类型和边索引维度是否匹配
                    if edge_type.size(0) != edge_index.size(1):
                        print(f"警告: 边类型维度 ({edge_type.size(0)}) 与边索引维度 ({edge_index.size(1)}) 不匹配，将调整边类型")
                        # 创建新的边类型向量，而不是警告
                        edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
                else:
                    # 如果没有边类型，创建默认值为0的边类型
                    edge_type = torch.zeros(edge_index.size(1), dtype=torch.long, device=edge_index.device)
                
                # 图层处理，添加异常处理
                try:
                    for layer in self.conv_layers:
                        features = self.relu(layer(features, edge_index, edge_type))
                except Exception as e:
                    print(f"图神经网络处理出错: {e}")
                    print(f"边索引形状: {edge_index.shape}, 边类型形状: {edge_type.shape}, 特征形状: {features.shape}")
            
            # 最终特征通过输出层
            features = self.dropout(self.relu(self.out_layer(features)))
            
            return features
    
    def extract_node_features(self, x):
        """
        提取节点特征的辅助方法
        """
        # 检查输入张量
        if x is None or x.shape[0] == 0:
            raise ValueError("输入特征为空或大小为0")
            
        # 确保LLM特征具有正确的维度
        llm_start = self.list_cat_num+self.list_numeric_num+self.list_tweet_channel+self.list_des_channel
        
        # 检查特征维度
        if x.shape[1] < llm_start:
            raise ValueError(f"特征维度不足: {x.shape[1]}, 需要至少 {llm_start}")
        
        # 分离特征
        cat_features = x[:, :self.list_cat_num]
        prop_features = x[:, self.list_cat_num: self.list_cat_num + self.list_numeric_num]
        tweet_features = x[:, self.list_cat_num+self.list_numeric_num: self.list_cat_num+self.list_numeric_num+self.list_tweet_channel]
        des_features = x[:, self.list_cat_num+self.list_numeric_num+self.list_tweet_channel: llm_start]
        llm_features = x[:, llm_start:]
        
        # 处理特征 - 添加维度检查
        features_numeric = self.dropout(self.relu(self.numeric_layer(prop_features)))
        features_bool = self.dropout(self.relu(self.cat_layer(cat_features)))
        features_tweet = self.dropout(self.relu(self.tweet_layer(tweet_features)))
        features_des = self.dropout(self.relu(self.des_layer(des_features)))
        features_llm = self.dropout(self.relu(self.llm_layer(llm_features)))
        
        # 检查特征维度是否一致
        if not (features_numeric.shape[1] == features_bool.shape[1] == features_tweet.shape[1] == 
                features_des.shape[1] == features_llm.shape[1] == self.embed_dim):
            raise ValueError(f"特征维度不一致: {features_numeric.shape[1]}, {features_bool.shape[1]}, "
                           f"{features_tweet.shape[1]}, {features_des.shape[1]}, {features_llm.shape[1]}")
        
        # 拼接特征
        try:
            features = torch.cat((features_numeric, features_bool, features_tweet, features_des, features_llm), dim=1)
            features = self.dropout(self.relu(self.hidden_layer(features)))
        except RuntimeError as e:
            print(f"特征拼接错误: {e}")
            print(f"特征形状: numeric={features_numeric.shape}, bool={features_bool.shape}, " 
                 f"tweet={features_tweet.shape}, des={features_des.shape}, llm={features_llm.shape}")
            raise
        
        return features 

    # 添加自定义日志方法，替代PyTorch Lightning内置的self.log
    def custom_log(self, name, value, prog_bar=False):
        """自定义日志方法，替代self.log"""
        self.log_dict[name] = value
        if prog_bar and hasattr(self, 'current_progress_bar') and self.current_progress_bar is not None:
            self.current_progress_bar.set_postfix({name: f"{value:.4f}"}) 