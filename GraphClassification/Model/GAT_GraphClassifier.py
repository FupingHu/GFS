import torch.nn as nn
import torch.nn.functional as F
from GraphClassification.Model.Layers import SpGraphAttentionLayer, pooling_sum_4tensor


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.gat1 = SpGraphAttentionLayer(in_features=input_dim, out_features=hidden_dim, dropout=0.1, alpha=0.2)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gat2 = SpGraphAttentionLayer(in_features=hidden_dim, out_features=hidden_dim, dropout=0.1, alpha=0.2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.gat3 = SpGraphAttentionLayer(in_features=hidden_dim, out_features=hidden_dim, dropout=0.1, alpha=0.2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator, labels):
        x = self.bn1(input_feature)
        x = self.gat1(input_feature, adjacency)
        x = self.bn2(x)
        x = self.gat2(x, adjacency)
        x = self.bn3(x)
        x = self.gat3(x, adjacency)
        x = pooling_sum_4tensor(x, graph_indicator, labels).cuda()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
