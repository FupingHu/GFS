import torch.nn as nn
import torch.nn.functional as F
from GraphClassification.Model.Layers import GraphSAGE, pooling_sum_4tensor


class GraphClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(GraphClassifier, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.gs1 = GraphSAGE(input_feat=input_dim, output_feat=hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gs2 = GraphSAGE(input_feat=hidden_dim, output_feat=hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.gs3 = GraphSAGE(input_feat=hidden_dim, output_feat=hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator, labels):
        x = self.bn1(input_feature)
        x = self.gs1(input_feature, adjacency)
        x = self.bn2(x)
        x = self.gs2(x, adjacency)
        x = self.bn3(x)
        x = self.gs3(x, adjacency)
        x = pooling_sum_4tensor(x, graph_indicator, labels).cuda()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
