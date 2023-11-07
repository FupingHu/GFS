import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import torch.nn.init as init


def pooling_mean_4tensor(features, ind, labels):
    res = []
    sum = len(labels)
    features = list(features.cpu().detach().numpy())
    for index in range(sum):
        temp = 0
        count = 0
        for idx, indicator in enumerate(ind):
            if index == indicator:
                temp += features[idx]
                count += 1
        if count == 0:
            count = 1
        res.append(temp / count)
    res = np.array(res)
    return torch.Tensor(res)


def pooling_sum_4tensor(features, ind, labels):
    res = []
    sum = len(labels)
    features = list(features.cpu().detach().numpy())
    for index in range(sum):
        temp = 0
        count = 0
        for idx, indicator in enumerate(ind):
            if index == indicator:
                temp += features[idx]
                count += 1
        if count == 0:
            count = 1
        res.append(temp)
    res = np.array(res)
    return torch.Tensor(res)


def pooling_max_4tensor(features, ind, labels):
    res = []
    sum = len(labels)
    features = list(features.cpu().detach().numpy())
    for index in range(sum):
        temp = 0
        for idx, indicator in enumerate(ind):
            if index == indicator:
                if (temp < features[idx]).any():
                    temp = features[idx]
        res.append(temp)
    res = np.array(res)
    return torch.Tensor(res)


def pooling_meanmax_4tensor(features, ind, labels):
    mean_out = pooling_mean_4tensor(features, ind, labels)
    max_out = pooling_max_4tensor(features, ind, labels)
    return torch.cat([mean_out, max_out], 1)


def pooling_summeanmax_4tensor(features, ind, labels):
    sum_out = pooling_sum_4tensor(features, ind, labels)
    mean_out = pooling_mean_4tensor(features, ind, labels)
    max_out = pooling_max_4tensor(features, ind, labels)
    return torch.cat([sum_out, mean_out, max_out], 1)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GraphConvolution_(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution_, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)

        assert not torch.isnan(h).any()

        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj):
        h = torch.mm(input, self.W)
        N = h.size()[0]

        a_input = torch.cat([h.repeat(1, N).view(N * N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2 * self.out_features)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, h)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GraphConvolution_MultiKernel(nn.Module):
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GraphConvolution_MultiKernel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """邻接矩阵是稀疏矩阵，因此在计算时使用稀疏矩阵乘法"""
        # torch.mm(mat1, mat2, out=None) → Tensor
        #     对矩阵mat1和mat2进行相乘。 如果mat1 是一个n×m张量，mat2 是一个 m×p 张量，将会输出一个 n×p 张量out。
        support = torch.mm(input_feature, self.weight)
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'


class GraphSAGE(nn.Module):

    def __init__(self, input_feat, output_feat, device="cuda:0", normalize=True):
        super(GraphSAGE, self).__init__()
        self.device = device
        self.normalize = normalize
        self.input_feat = input_feat
        self.output_feat = output_feat
        self.linear = nn.Linear(self.input_feat, self.output_feat)
        self.layer_norm = nn.LayerNorm(self.output_feat)  # elementwise_affine=False
        nn.init.xavier_uniform_(self.linear.weight)

    def aggregate_convolutional(self, x, a):
        eye = torch.eye(a.shape[0], dtype=torch.float, device=self.device)
        a = a + eye
        h_hat = a @ x

        return h_hat

    def forward(self, x, a):
        h_hat = self.aggregate_convolutional(x, a)
        h = F.relu(self.linear(h_hat))
        if self.normalize:
            # h = F.normalize(h, p=2, dim=1)  # Normalize edge embeddings
            h = self.layer_norm(h)  # Normalize layerwise (mean=0, std=1)

        return h


class DiffPool(nn.Module):

    def __init__(self, feature_size, output_dim, device="cuda:0", final_layer=False):
        super(DiffPool, self).__init__()
        self.device = device
        self.feature_size = feature_size
        self.output_dim = output_dim
        self.embed = GraphSAGE(self.feature_size, self.feature_size, device=self.device)
        self.pool = GraphSAGE(self.feature_size, self.output_dim, device=self.device)
        self.final_layer = final_layer

    def forward(self, x, a):
        z = self.embed(x, a)
        if self.final_layer:
            s = torch.ones(x.size(0), self.output_dim, device=self.device)
        else:
            s = F.softmax(self.pool(x, a), dim=1)
        x_new = s.t() @ z
        a_new = s.t() @ a @ s
        return x_new, a_new


class AdaGCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, dropout_adj):
        super(AdaGCN, self).__init__()
        fcs = [MixedLinear(nfeat, nhid, bias=False)]
        fcs.append(nn.Linear(nhid, nclass, bias=False))
        self.fcs = nn.ModuleList(fcs)
        self.reg_params = list(self.fcs[0].parameters())

        if dropout is 0:
            self.dropout = lambda x: x
        else:
            self.dropout = MixedDropout(dropout)  # p: drop rate
        if dropout_adj is 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj)  # p: drop rate
        self.act_fn = nn.ReLU()

    def _transform_features(self, x):
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))
        return res

    def forward(self, x, adj, idx):  # X, A
        logits = self._transform_features(x)  # MLP: X->H, Mixed-layer + some layers FC
        return F.log_softmax(logits, dim=-1)[idx]

class SparseDropout(nn.Module):
    def __init__(self, p):
        super(SparseDropout, self).__init__()
        self.p = p

    def forward(self, input):
        input_coal = input.coalesce()
        drop_val = F.dropout(input_coal._values(), self.p, self.training)
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)

class MixedDropout(nn.Module):
    def __init__(self, p):
        super(MixedDropout, self).__init__()
        self.dense_dropout = nn.Dropout(p)
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        if input.is_sparse:
            return self.sparse_dropout(input)
        else:
            return self.dense_dropout(input)


class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MixedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Our fan_in is interpreted by PyTorch as fan_out (swapped dimensions)
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))
        if self.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.bias is None:
            if input.is_sparse:
                res = torch.sparse.mm(input, self.weight)
            else:
                res = input.matmul(self.weight)
        else:
            if input.is_sparse:
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1), input, self.weight)
            else:
                res = torch.addmm(self.bias, input, self.weight)
        return res

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)