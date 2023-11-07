import numpy as np
import torch
from grakel import PropagationAttr
from grakel.datasets import fetch_dataset
from scipy.special import softmax
from sklearn import preprocessing
import scipy


def GraphStruct2XandA(edges, node_features):
    node_key_list = list(node_features.keys())
    X = list()
    for node_key in node_features.keys():
        X.append(node_features[node_key])
    A = np.zeros(shape=(len(X), len(X)))
    for edge in edges:
        A[edge[0] - node_key_list[0]][edge[1] - node_key_list[0]] = 1
    return np.array(X), np.array(A)


def tensor_from_numpy(x, device):
    return torch.from_numpy(x).to(device)


def Preprocess_GandMUandY(minibatch_X, minibatch_Mu, minibatch_Y):
    NodeFeatures_Train = list()
    AdjMat_Train = list()
    mu4train_list = softmax(minibatch_Mu, axis=1)
    mu4train_list = tensor_from_numpy(mu4train_list, 'cuda')

    OriginTrainData = minibatch_X
    min_max_scaler = preprocessing.MinMaxScaler()
    for originTrainData in OriginTrainData:
        X_train, A_train = GraphStruct2XandA(originTrainData[0], originTrainData[1])
        X_train_normal = min_max_scaler.fit_transform(X_train)
        NodeFeatures_Train.append(X_train_normal)
        AdjMat_Train.append(A_train)

    A_all = AdjMat_Train[0]
    for id, adj in enumerate(AdjMat_Train):
        if id != 0:
            A_all = scipy.linalg.block_diag(A_all, adj)

    X_all = list()
    Ind = list()
    for id, X in enumerate(NodeFeatures_Train):
        for X_feature in X:
            X_all.append(X_feature)
            Ind.append(id)
    X_all = np.array(X_all)
    Ind = np.array(Ind)

    I = np.array(np.eye(A_all.shape[0]))
    A_all_hat = A_all + I
    D_hat = np.sum(A_all_hat, axis=0)
    D_hat = np.diag(D_hat ** -0.5)

    y_train = np.array(minibatch_Y)
    D_hat = torch.Tensor(D_hat).cuda()
    A_all_hat = torch.Tensor(A_all_hat).cuda()
    TempMat = torch.mm(D_hat, A_all_hat)
    InputAdjHat = torch.mm(TempMat, D_hat)
    input_dim = X_all.shape[1]
    return InputAdjHat, Ind, y_train, input_dim, X_all, mu4train_list


def GetSim(G):
    gk = PropagationAttr(normalize=True)
    K = gk.fit_transform(G)
    return K[1][0]
