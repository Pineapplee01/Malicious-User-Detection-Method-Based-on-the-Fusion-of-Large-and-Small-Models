import torch
from torch_geometric.data import Data
import os

def load_data_llm(args):
    # 确保路径存在
    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"数据路径不存在: {args.dataset_path}")
    
    # 设置默认参数（如果缺失）
    if not hasattr(args, 'pretext_task'):
        print("警告: 未设置pretext_task参数，使用默认值'contrastive'")
        args.pretext_task = "contrastive"
        
    if not hasattr(args, 'template'):
        print("警告: 未设置template参数，使用默认值'l'")
        args.template = "l"
        
    if not hasattr(args, 'list_num'):
        print("警告: 未设置list_num参数，使用默认值50")
        args.list_num = 50
        
    if not hasattr(args, 'edge_types'):
        print("警告: 未设置edge_types参数，使用默认值'all'")
        args.edge_types = "all"
        
    if not hasattr(args, 'llm_enhancement_channel'):
        print("警告: 未设置llm_enhancement_channel参数，使用默认值100")
        args.llm_enhancement_channel = 100
    
    print("加载用户基础特征...")
    try:
        user_cat_features = torch.load(os.path.join(args.dataset_path, "user_cat_properties_tensor.pt"), map_location='cpu')
        user_prop_features = torch.load(os.path.join(args.dataset_path, "user_num_properties_tensor.pt"), map_location='cpu')
        user_tweet_features = torch.load(os.path.join(args.dataset_path, "user_tweets_tensor.pt"), map_location='cpu')
        user_des_features = torch.load(os.path.join(args.dataset_path, "user_des_tensor.pt"), map_location='cpu')
    except Exception as e:
        raise FileNotFoundError(f"加载用户基础特征失败: {e}\n请确认已运行预处理脚本")
    
    # 加载LLM增强的用户特征
    print(f"加载用户LLM增强特征 ({args.llm_model})...")
    user_llm_file = os.path.join(args.dataset_path, f"user_llm_features_{args.llm_model}.pt")
    if os.path.exists(user_llm_file):
        user_llm_features = torch.load(user_llm_file, map_location='cpu')
    else:
        print(f"警告: 未找到用户LLM特征 {user_llm_file}，使用零填充")
        user_llm_features = torch.zeros((user_cat_features.shape[0], args.llm_enhancement_channel), dtype=torch.float)
    
    # 拼接所有用户特征
    user_feature_dim = user_cat_features.shape[1] + user_prop_features.shape[1] + user_tweet_features.shape[1] + user_des_features.shape[1] + user_llm_features.shape[1]
    user_x = torch.cat((user_cat_features, user_prop_features, user_tweet_features, user_des_features, user_llm_features), dim=1)
    
    # 加载列表特征（如果启用）
    list_x = None
    if args.lst:
        print("加载列表基础特征...")
        try:
            list_cat_features = torch.load(os.path.join(args.dataset_path, "list_cat_properties_tensor.pt"), map_location='cpu')
            list_prop_features = torch.load(os.path.join(args.dataset_path, "list_num_properties_tensor.pt"), map_location='cpu')
            list_tweet_features = torch.load(os.path.join(args.dataset_path, "list_tweets_tensor.pt"), map_location='cpu')
            list_des_features = torch.load(os.path.join(args.dataset_path, "list_des_tensor.pt"), map_location='cpu')
        except Exception as e:
            raise FileNotFoundError(f"加载列表基础特征失败: {e}\n请确认已运行预处理脚本")
        
        # 加载LLM增强的列表特征
        list_llm_file = os.path.join(args.dataset_path, f"list_llm_features_{args.llm_model}.pt")
        if os.path.exists(list_llm_file):
            list_llm_features = torch.load(list_llm_file, map_location='cpu')
        else:
            print(f"警告: 未找到列表LLM特征 {list_llm_file}，使用零填充")
            list_llm_features = torch.zeros((list_cat_features.shape[0], args.llm_enhancement_channel), dtype=torch.float)
        
        # 打印列表特征维度
        print("列表张量形状检查:")
        print(f"  list_cat_features: {list_cat_features.shape}")
        print(f"  list_prop_features: {list_prop_features.shape}")
        print(f"  list_tweet_features: {list_tweet_features.shape}")
        print(f"  list_des_features: {list_des_features.shape}")
        print(f"  list_llm_features: {list_llm_features.shape}")
        
        # 检查列表特征总维度是否与用户特征维度匹配
        list_feature_dim = list_cat_features.shape[1] + list_prop_features.shape[1] + list_tweet_features.shape[1] + list_des_features.shape[1] + list_llm_features.shape[1]
        
        # 如果维度不匹配，添加零填充以匹配用户特征维度
        if list_feature_dim != user_feature_dim:
            padding_size = abs(user_feature_dim - list_feature_dim)
            print(f"特征维度不匹配: 用户={user_feature_dim}, 列表={list_feature_dim}, 差距={padding_size}")
            
            if list_feature_dim < user_feature_dim:
                # 列表特征维度小于用户特征维度，添加零填充
                list_padding = torch.zeros((list_cat_features.shape[0], padding_size), dtype=torch.float)
                print(f"创建列表填充向量，形状: {list_padding.shape}")
                print(f"  list_padding: {list_padding.shape}")
                # 拼接所有列表特征
                list_x = torch.cat((list_cat_features, list_prop_features, list_tweet_features, list_des_features, list_llm_features, list_padding), dim=1)
            else:
                # 用户特征维度小于列表特征维度，添加零填充到用户特征
                user_padding = torch.zeros((user_cat_features.shape[0], padding_size), dtype=torch.float)
                print(f"创建用户填充向量，形状: {user_padding.shape}")
                # 重新拼接用户特征
                user_x = torch.cat((user_cat_features, user_prop_features, user_tweet_features, user_des_features, user_llm_features, user_padding), dim=1)
                # 拼接所有列表特征
                list_x = torch.cat((list_cat_features, list_prop_features, list_tweet_features, list_des_features, list_llm_features), dim=1)
        else:
            # 维度匹配，直接拼接
            list_x = torch.cat((list_cat_features, list_prop_features, list_tweet_features, list_des_features, list_llm_features), dim=1)
    
    # 打印最终特征维度
    print(f"用户特征形状: {user_x.shape}")
    if list_x is not None:
        print(f"列表特征形状: {list_x.shape}")
        # 确认两者维度一致
        assert user_x.shape[1] == list_x.shape[1], f"特征维度不匹配: 用户={user_x.shape[1]}, 列表={list_x.shape[1]}"
    
    print("加载预训练标签和索引...")
    try:
        if args.pretext_task == "contrastive":
            if args.template in ["l", "s"]:
                pretrain_label = torch.load(os.path.join(args.dataset_path, "pretrain_labels_index.pt"), map_location='cpu')
            else:
                pretrain_label = torch.load(os.path.join(args.dataset_path, f"pretrain_labels_index_{args.template}.pt"), map_location='cpu')
        elif args.pretext_task == "multi":
            if os.path.exists(os.path.join(args.dataset_path, "pretrain_labels_index_multi.pt")):
                pretrain_label = torch.load(os.path.join(args.dataset_path, "pretrain_labels_index_multi.pt"), map_location='cpu')
            else:
                print("警告: 未找到multi预训练标签，使用普通预训练标签代替")
                pretrain_label = torch.load(os.path.join(args.dataset_path, "pretrain_labels_index.pt"), map_location='cpu')
    except Exception as e:
        raise FileNotFoundError(f"加载预训练标签失败: {e}\n请确认已运行预处理脚本")
    
    print("加载微调标签和索引...")
    try:
        finetune_label = torch.load(os.path.join(args.dataset_path, "finetune_label.pt"), map_location='cpu')
    except Exception as e:
        raise FileNotFoundError(f"加载微调标签失败: {e}\n请确认已运行预处理脚本")
    
    if args.lst:
        all_x = torch.cat((user_x, list_x), dim=0)
        print("加载用户和列表边...")
        edge_path = os.path.join(args.dataset_path, f"edge_index(ul_{args.edge_types}).pt")
        edge_type_path = os.path.join(args.dataset_path, f"edge_type(ul_{args.edge_types}).pt")
        
        # 尝试加载指定边，如果失败则尝试加载基本边
        try:
            edge_index = torch.load(edge_path, map_location='cpu')
            edge_type = torch.load(edge_type_path, map_location='cpu').unsqueeze(-1)
        except Exception as e:
            print(f"加载指定边失败: {e}，尝试加载基本边")
            edge_index = torch.load(os.path.join(args.dataset_path, "edge_index.pt"), map_location='cpu')
            edge_type = torch.load(os.path.join(args.dataset_path, "edge_type.pt"), map_location='cpu').unsqueeze(-1)
            
        # 为列表节点添加-1标签
        lst_label_padding = torch.full((args.list_num,), -1)
        if args.pretext_task == "multi":
            if pretrain_label.dim() > 1 and pretrain_label.shape[1] == 153:
                pre_lst_label_padding = torch.full((args.list_num, 153), -1)
                pretrain_label = torch.cat((pretrain_label, pre_lst_label_padding))
            else:
                pretrain_label = torch.cat((pretrain_label, lst_label_padding))
        else:
            pretrain_label = torch.cat((pretrain_label, lst_label_padding))
        finetune_label = torch.cat((finetune_label, lst_label_padding))
    else:
        all_x = user_x
        print("加载用户边...")
        try:
            edge_index = torch.load(os.path.join(args.dataset_path, "edge_index.pt"), map_location='cpu')
            edge_type = torch.load(os.path.join(args.dataset_path, "edge_type.pt"), map_location='cpu').unsqueeze(-1)
        except Exception as e:
            print(f"警告: 加载边结构失败: {e}，将创建空边集")
            # 创建空边集
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_type = torch.zeros((0, 1), dtype=torch.long)
    
    print("加载训练、验证、测试索引...")
    try:
        pretrain_train_idx = torch.load(os.path.join(args.dataset_path, "pretrain_train_idx.pt"), map_location='cpu')
        pretrain_valid_idx = torch.load(os.path.join(args.dataset_path, "pretrain_val_idx.pt"), map_location='cpu')
        pretrain_test_idx = torch.load(os.path.join(args.dataset_path, "pretrain_test_idx.pt"), map_location='cpu')
        finetune_train_idx = torch.load(os.path.join(args.dataset_path, "finetune_train_idx.pt"), map_location='cpu')
        finetune_valid_idx = torch.load(os.path.join(args.dataset_path, "finetune_val_idx.pt"), map_location='cpu')
        finetune_test_idx = torch.load(os.path.join(args.dataset_path, "finetune_test_idx.pt"), map_location='cpu')
    except Exception as e:
        raise FileNotFoundError(f"加载训练、验证、测试索引失败: {e}\n请确认已运行预处理脚本")
        
    print("创建数据对象...")
    # 创建数据对象
    pretrain_data = Data(x=all_x, edge_index=edge_index, edge_attr=edge_type, y=pretrain_label, finetune_label=finetune_label)
    pretrain_data.pretrain_train_idx = pretrain_train_idx
    pretrain_data.pretrain_valid_idx = pretrain_valid_idx
    pretrain_data.pretrain_test_idx = pretrain_test_idx
    pretrain_data.n_id = torch.arange(pretrain_data.num_nodes)
    
    # 创建训练/验证/测试掩码
    train_mask = torch.zeros(pretrain_data.num_nodes, dtype=torch.bool, device=pretrain_data.x.device)
    train_mask[pretrain_train_idx.long()] = True
    pretrain_data.train_mask = train_mask
    
    val_mask = torch.zeros(pretrain_data.num_nodes, dtype=torch.bool, device=pretrain_data.x.device)
    val_mask[pretrain_valid_idx.long()] = True
    pretrain_data.val_mask = val_mask
    
    test_mask = torch.zeros(pretrain_data.num_nodes, dtype=torch.bool, device=pretrain_data.x.device)
    test_mask[pretrain_test_idx.long()] = True
    pretrain_data.test_mask = test_mask

    finetune_data = Data(x=all_x, edge_index=edge_index, edge_attr=edge_type, y=finetune_label, pretrain_label=pretrain_label)
    finetune_data.finetune_train_idx = finetune_train_idx
    finetune_data.finetune_valid_idx = finetune_valid_idx
    finetune_data.finetune_test_idx = finetune_test_idx
    finetune_data.n_id = torch.arange(finetune_data.num_nodes)
    
    # 创建训练/验证/测试掩码
    train_mask = torch.zeros(finetune_data.num_nodes, dtype=torch.bool, device=finetune_data.x.device)
    train_mask[finetune_train_idx.long()] = True
    finetune_data.train_mask = train_mask
    
    val_mask = torch.zeros(finetune_data.num_nodes, dtype=torch.bool, device=finetune_data.x.device)
    val_mask[finetune_valid_idx.long()] = True
    finetune_data.val_mask = val_mask
    
    test_mask = torch.zeros(finetune_data.num_nodes, dtype=torch.bool, device=finetune_data.x.device)
    test_mask[finetune_test_idx.long()] = True
    finetune_data.test_mask = test_mask
    
    print(f"数据加载完成，节点数量: {pretrain_data.num_nodes}, 边数量: {pretrain_data.edge_index.shape[1]}")

    return pretrain_data, finetune_data 
