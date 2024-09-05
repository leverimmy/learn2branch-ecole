import gzip
import pickle
import datetime
import ecole
import numpy as np

import torch
import torch.nn.functional as F
import torch_geometric


def log(str, logfile=None):
    str = f'[{datetime.datetime.now()}] {str}'
    print(str)
    if logfile is not None:
        with open(logfile, mode='a') as f:
            print(str, file=f)


def pad_tensor(input_, pad_sizes, pad_value=-1e8):
    max_pad_size = pad_sizes.max()
    output = input_.split(pad_sizes.cpu().numpy().tolist())
    output = torch.stack([F.pad(slice_, (0, max_pad_size-slice_.size(0)), 'constant', pad_value)
                          for slice_ in output], dim=0)
    return output


class BipartiteNodeData(torch_geometric.data.Data):
    def __init__(self, constraint_features, edge_indices, edge_features, variable_features,
                 candidates, nb_candidates, candidate_choice, candidate_scores):
        super().__init__()
        self.constraint_features = constraint_features
        self.edge_index = edge_indices
        self.edge_attr = edge_features
        self.variable_features = variable_features
        self.candidates = candidates
        self.nb_candidates = nb_candidates
        self.candidate_choices = candidate_choice
        self.candidate_scores = candidate_scores

    def __inc__(self, key, value, store, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.constraint_features.size(0)], [self.variable_features.size(0)]])
        elif key == 'candidates':
            return self.variable_features.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class GraphDataset(torch_geometric.data.Dataset):
    def __init__(self, sample_files):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.sample_files = sample_files

    def len(self):
        return len(self.sample_files)

    def get(self, index):
        with gzip.open(self.sample_files[index], 'rb') as f:
            sample = pickle.load(f)

        sample_observation, sample_action, sample_action_set, sample_scores = sample['data']

        constraint_features, (edge_indices, edge_features), variable_features = sample_observation
        constraint_features = torch.FloatTensor(constraint_features)
        edge_indices = torch.LongTensor(edge_indices.astype(np.int32))
        edge_features = torch.FloatTensor(np.expand_dims(edge_features, axis=-1))
        variable_features = torch.FloatTensor(variable_features)

        candidates = torch.LongTensor(np.array(sample_action_set, dtype=np.int32))
        candidate_choice = torch.where(candidates == sample_action)[0][0]  # action index relative to candidates
        candidate_scores = torch.FloatTensor([sample_scores[j] for j in candidates])

        graph = BipartiteNodeData(constraint_features, edge_indices, edge_features, variable_features,
                                  candidates, len(candidates), candidate_choice, candidate_scores)
        graph.num_nodes = constraint_features.shape[0]+variable_features.shape[0]
        return graph


class Scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer, **kwargs):
        super().__init__(optimizer, **kwargs)

    def step(self, metrics):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        self.last_epoch =+1

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs == self.patience:
            self._reduce_lr(self.last_epoch)

        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]


class NodeTripartite(ecole.observation.NodeBipartite):
    def extract(self, model, done):
        # 调用父类的 extract 方法获取二部图信息
        observation = super().extract(model, done)

        # 解构得到的二部图信息
        row_features = observation.row_features
        variable_features = observation.variable_features
        edge_features = observation.edge_features
        
        # print(f'[原来]indices size = {len(edge_features.indices)}\nvalues size = {len(edge_features.values)}')
        # print(edge_features.indices)
        # print(edge_features.values)

        # 添加目标节点特征
        objective_feature = variable_features[-1] # 定义目标节点的特征
        num_variables = variable_features.shape[0]
        num_constraints = row_features.shape[0]
        objective_index = num_variables + num_constraints

        # 将目标节点特征加入变量特征
        variable_features = np.vstack([variable_features, objective_feature])

        # 创建新的边索引，将目标节点连接到所有其他节点
        new_edges = []
        
        # 连接目标节点到所有变量节点
        for i in range(num_variables):
            new_edges.append([objective_index, i])
            new_edges.append([i, objective_index])

        # 连接目标节点到所有约束节点
        for i in range(num_constraints):
            constraint_node_index = num_variables + i
            new_edges.append([objective_index, constraint_node_index])
            new_edges.append([constraint_node_index, objective_index])

        # 合并现有边和新的边
        new_edge_indices = np.array(new_edges).T
        edge_features.indices = np.hstack([edge_features.indices, new_edge_indices])
        
        # print(f'[现在]indices size = {len(edge_features.indices)}\nvalues size = {len(edge_features.values)}')
        # print(edge_features.indices)
        # print(edge_features.values)

        # 返回新的三部图观察
        return row_features, edge_features, variable_features
