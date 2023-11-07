import time

import numpy as np
from grakel import PropagationAttr, ShortestPathAttr


def GetSim(G):
    # Get the graph kernel matrix
    gk = ShortestPathAttr(normalize=True)
    K = gk.fit_transform(G)
    return K[1][0]


def GetSim_WithoutNodeAttr(G):
    # Get the graph kernel matrix
    gk = ShortestPathAttr(normalize=True)
    K = gk.fit_transform(G)
    return K[1][0]


def GetAnt(G_train, G_test, Rules):
    # Step-1 Initialize the cluster prototype graph
    InitProtGraphIds = np.random.randint(low=0, high=len(G_train), size=Rules)
    PrototypeGraphs = list()
    for id, val in enumerate(InitProtGraphIds):
        PrototypeGraphs.append(G_train[val])

    Cluster = list()
    for i in range(Rules):
        Cluster.append([])
        # exec('Cluser{} = {}'.format(i, []))

    for i, g in enumerate(G_train):
        CompareGraph = list()
        for j in range(Rules):
            CompareGraph.append([PrototypeGraphs[j], g])
            # exec('CompareGraph{} = {}'.format(j, [PrototypeGraphs[j], g]))

        Sim = list()
        Sims = list()
        for k in range(Rules):
            Sim.append(GetSim(CompareGraph[k]))
            # exec('Sim{} = GetSim(CompareGraph{})'.format(k, k))

        for l in range(Rules):
            Sims.append(Sim[l])
            # exec('Sims.append(Sim{})'.format(l))

        maxsim = max(Sims)
        label = Sims.index(max(Sims))
        Cluster[label].append(g)
        # exec('Cluser{}.append(g)'.format(label))

    SumSimCluster = list()
    for K in range(Rules):
        SumSimCluster.append([])
        for i in range(len(Cluster[K])):
            SumSim = 0
            for j in range(len(Cluster[K])):
                temp_compare_graphs = [Cluster[K][i], Cluster[K][j]]
                temp_sim = GetSim(temp_compare_graphs)
                SumSim += temp_sim
                SumSimCluster[K].append(SumSim)
        # exec(
        #     'SumSimCluster{} = []\nfor i in range(len(Cluser{})):\n    SumSim = 0\nfor j in range(len(Cluser{})):\n    temp_compare_graphs = [Cluser{}[i], Cluser{}[j]]\ntemp_sim = GetSim(temp_compare_graphs)\nSumSim += temp_sim\nSumSimCluster{}.append(SumSim)'.format(
        #         K, K, K, K, K, K))

    MaxIndex_Cluster = list()
    MaxOfSumSimCluster = list()
    for K in range(Rules):
        MaxIndex_Cluster[K] = SumSimCluster[K].index(max(SumSimCluster[K]))
        MaxOfSumSimCluster[K] = max(SumSimCluster[K])
        # exec(
        #     'MaxIndex_Cluster{} = SumSimCluster{}.index(max(SumSimCluster{}))\nMaxOfSumSimCluster{} = max(SumSimCluster{})'.format(
        #         K, K, K, K, K))

    ClusterCenters = list()
    for K in range(Rules):
        ClusterCenters.append(Cluster[K][MaxIndex_Cluster[K]])
        # exec('ClusterCenters.append(Cluser{}[MaxIndex_Cluster{}])'.format(K, K))

    # Compute normal membership of training data
    Mu4Train_list = []
    for i, g in enumerate(G_train):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Train_list.append(mu)
    Mu4Train_list_Result = np.array(Mu4Train_list)

    # 计算测试数据的归一化隶属度
    Mu4Test_list = []
    for i, g in enumerate(G_test):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Test_list.append(mu)
    Mu4Test_list_Result = np.array(Mu4Test_list)
    return ClusterCenters, Mu4Train_list_Result, Mu4Test_list_Result

def GetSim_(G):
    start_time = time.time()
    # Get the graph kernel matrix
    gk = ShortestPathAttr(normalize=True)
    K = gk.fit_transform(G)
    end_time = time.time()
    print('Kernel runtime:{}'.format(end_time-start_time))
    return K[1][0]

def GetAntWithVal_(G_train, G_val, G_test, Rules):
    # Step-1 Initialize the cluster prototype graph
    InitProtGraphIds = np.random.randint(low=0, high=len(G_train), size=Rules)
    PrototypeGraphs = []
    for id, val in enumerate(InitProtGraphIds):
        PrototypeGraphs.append(G_train[val])

    Cluster = list()
    for i in range(Rules):
        Cluster.append([])
        # exec('Cluser{} = {}'.format(i, []))

    for i, g in enumerate(G_train):
        CompareGraph = list()
        for j in range(Rules):
            CompareGraph.append([PrototypeGraphs[j], g])
            # exec('CompareGraph{} = {}'.format(j, [PrototypeGraphs[j], g]))
        Sim = list()
        Sims = list()
        for k in range(Rules):
            Sim.append([])
            Sim[k] = GetSim_(CompareGraph[k])
            # exec('Sim{} = GetSim(CompareGraph{})'.format(k, k))

        for l in range(Rules):
            Sims.append(Sim[l])
            # exec('Sims.append(Sim{})'.format(l))

        maxsim = max(Sims)
        label = Sims.index(max(Sims))
        Cluster[label].append(g)
        # exec('Cluser{}.append(g)'.format(label))

    SumSimCluster = list()
    for K in range(Rules):
        SumSimCluster.append([])
        for i in range(len(Cluster[K])):
            SumSim = 0
            for j in range(len(Cluster[K])):
                temp_compare_graphs = [Cluster[K][i], Cluster[K][j]]
                temp_sim = GetSim_(temp_compare_graphs)
                SumSim += temp_sim
            SumSimCluster[K].append(SumSim)
        # exec(
        #     'SumSimCluster{} = []\nfor i in range(len(Cluser{})):\n    SumSim = 0\nfor j in range(len(Cluser{})):\n    temp_compare_graphs = [Cluser{}[i], Cluser{}[j]]\ntemp_sim = GetSim(temp_compare_graphs)\nSumSim += temp_sim\nSumSimCluster{}.append(SumSim)'.format(
        #         K, K, K, K, K, K))

    MaxIndex_Cluster = list()
    MaxOfSumSimCluster = list()
    for K in range(Rules):
        # Init
        MaxIndex_Cluster.append(0)
        MaxOfSumSimCluster.append(0)

        MaxIndex_Cluster[K] = SumSimCluster[K].index(max(SumSimCluster[K]))
        MaxOfSumSimCluster[K] = max(SumSimCluster[K])
        # exec(
        #     'MaxIndex_Cluster{} = SumSimCluster{}.index(max(SumSimCluster{}))\nMaxOfSumSimCluster{} = max(SumSimCluster{})'.format(
        #         K, K, K, K, K))
    ClusterCenters = list()
    for K in range(Rules):
        ClusterCenters.append(Cluster[K][MaxIndex_Cluster[K]])
        # exec('ClusterCenters.append(Cluser{}[MaxIndex_Cluster{}])'.format(K, K))

    # 计算训练数据的归一化隶属度
    Mu4Train_list = []
    for i, g in enumerate(G_train):
        FireLevel_Sum = 0
        FireLevel_List = list()
        mu = list()
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim_(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Train_list.append(mu)
    Mu4Train_list_Result = np.array(Mu4Train_list)

    # 计算验证数据的归一化隶属度
    Mu4Val_list = []
    for i, g in enumerate(G_val):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Val_list.append(mu)
    Mu4Val_list_Result = np.array(Mu4Val_list)

    # 计算测试数据的归一化隶属度
    Mu4Test_list = []
    for i, g in enumerate(G_test):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Test_list.append(mu)
    Mu4Test_list_Result = np.array(Mu4Test_list)
    return ClusterCenters, Mu4Train_list_Result, Mu4Val_list_Result, Mu4Test_list_Result


# 2023-02-13
def GetAntWithVal(G_train, G_val, G_test, Rules):
    # Step-1 Initialize the cluster prototype graph
    InitProtGraphIds = np.random.randint(low=0, high=len(G_train), size=Rules)
    PrototypeGraphs = list()
    for id, val in enumerate(InitProtGraphIds):
        PrototypeGraphs.append(G_train[val])

    Cluster = list()
    for i in range(Rules):
        Cluster.append([])
        # exec('Cluser{} = {}'.format(i, []))

    for i, g in enumerate(G_train):
        CompareGraph = list()
        for j in range(Rules):
            CompareGraph.append([PrototypeGraphs[j], g])
            # exec('CompareGraph{} = {}'.format(j, [PrototypeGraphs[j], g]))
        Sim = list()
        Sims = list()
        for k in range(Rules):
            Sim.append([])
            Sim[k] = GetSim(CompareGraph[k])
            # exec('Sim{} = GetSim(CompareGraph{})'.format(k, k))

        for l in range(Rules):
            Sims.append(Sim[l])
            # exec('Sims.append(Sim{})'.format(l))

        maxsim = max(Sims)
        label = Sims.index(max(Sims))
        Cluster[label].append(g)
        # exec('Cluser{}.append(g)'.format(label))

    SumSimCluster = list()
    for K in range(Rules):
        SumSimCluster.append([])
        for i in range(len(Cluster[K])):
            SumSim = 0
            for j in range(len(Cluster[K])):
                temp_compare_graphs = [Cluster[K][i], Cluster[K][j]]
                temp_sim = GetSim(temp_compare_graphs)
                SumSim += temp_sim
            SumSimCluster[K].append(SumSim)
        # exec(
        #     'SumSimCluster{} = []\nfor i in range(len(Cluser{})):\n    SumSim = 0\nfor j in range(len(Cluser{})):\n    temp_compare_graphs = [Cluser{}[i], Cluser{}[j]]\ntemp_sim = GetSim(temp_compare_graphs)\nSumSim += temp_sim\nSumSimCluster{}.append(SumSim)'.format(
        #         K, K, K, K, K, K))

    MaxIndex_Cluster = list()
    MaxOfSumSimCluster = list()
    for K in range(Rules):
        # Init
        MaxIndex_Cluster.append(0)
        MaxOfSumSimCluster.append(0)

        MaxIndex_Cluster[K] = SumSimCluster[K].index(max(SumSimCluster[K]))
        MaxOfSumSimCluster[K] = max(SumSimCluster[K])
        # exec(
        #     'MaxIndex_Cluster{} = SumSimCluster{}.index(max(SumSimCluster{}))\nMaxOfSumSimCluster{} = max(SumSimCluster{})'.format(
        #         K, K, K, K, K))
    ClusterCenters = list()
    for K in range(Rules):
        print(MaxIndex_Cluster[K])
        print(len(Cluster))
        print(len(Cluster[K]))
        ClusterCenters.append(Cluster[K][MaxIndex_Cluster[K]])
        # exec('ClusterCenters.append(Cluser{}[MaxIndex_Cluster{}])'.format(K, K))

    # 计算训练数据的归一化隶属度
    Mu4Train_list = []
    for i, g in enumerate(G_train):
        FireLevel_Sum = 0
        FireLevel_List = list()
        mu = list()
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Train_list.append(mu)
    Mu4Train_list_Result = np.array(Mu4Train_list)

    # 计算验证数据的归一化隶属度
    Mu4Val_list = []
    for i, g in enumerate(G_val):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Val_list.append(mu)
    Mu4Val_list_Result = np.array(Mu4Val_list)

    # 计算测试数据的归一化隶属度
    Mu4Test_list = []
    for i, g in enumerate(G_test):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Test_list.append(mu)
    Mu4Test_list_Result = np.array(Mu4Test_list)
    return ClusterCenters, Mu4Train_list_Result, Mu4Val_list_Result, Mu4Test_list_Result


def GetAntWithVal_WithoutNodeAttr(G_train, G_val, G_test, Rules):
    # Step-1 Initialize the cluster prototype graph
    InitProtGraphIds = np.random.randint(low=0, high=len(G_train), size=Rules)
    PrototypeGraphs = []
    for id, val in enumerate(InitProtGraphIds):
        PrototypeGraphs.append(G_train[val])


    for i in range(Rules):
        exec('Cluser{} = {}'.format(i, []))

    for i, g in enumerate(G_train):
        for j in range(Rules):
            exec('CompareGraph{} = {}'.format(j, [PrototypeGraphs[j], g]))
        Sims = []
        for k in range(Rules):
            exec('Sim{} = GetSim_WithoutNodeAttr(CompareGraph{})'.format(k, k))

        for l in range(Rules):
            exec('Sims.append(Sim{})'.format(l))

        maxsim = max(Sims)
        label = Sims.index(max(Sims))
        exec('Cluser{}.append(g)'.format(label))

    for K in range(Rules):
        exec(
            'SumSimCluster{} = []\nfor i in range(len(Cluser{})):\n    SumSim = 0\nfor j in range(len(Cluser{})):\n    temp_compare_graphs = [Cluser{}[i], Cluser{}[j]]\ntemp_sim = GetSim_WithoutNodeAttr(temp_compare_graphs)\nSumSim += temp_sim\nSumSimCluster{}.append(SumSim)'.format(
                K, K, K, K, K, K))

    for K in range(Rules):
        exec(
            'MaxIndex_Cluster{} = SumSimCluster{}.index(max(SumSimCluster{}))\nMaxOfSumSimCluster{} = max(SumSimCluster{})'.format(
                K, K, K, K, K))
    ClusterCenters = []
    for K in range(Rules):
        exec('ClusterCenters.append(Cluser{}[MaxIndex_Cluster{}])'.format(K, K))

    # 计算训练数据的归一化隶属度
    Mu4Train_list = []
    for i, g in enumerate(G_train):
        FireLevel_Sum = 0
        FireLevel_List = list()
        mu = list()
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim_WithoutNodeAttr(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Train_list.append(mu)
    Mu4Train_list_Result = np.array(Mu4Train_list)

    # 计算验证数据的归一化隶属度
    Mu4Val_list = []
    for i, g in enumerate(G_val):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim_WithoutNodeAttr(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Val_list.append(mu)
    Mu4Val_list_Result = np.array(Mu4Val_list)

    # 计算测试数据的归一化隶属度
    Mu4Test_list = []
    for i, g in enumerate(G_test):
        FireLevel_Sum = 0
        FireLevel_List = []
        mu = []
        for k in range(Rules):
            computed_graphs = [ClusterCenters[k], g]
            Temp_Sim = GetSim_WithoutNodeAttr(computed_graphs)
            FireLevel_Sum += Temp_Sim
            FireLevel_List.append(Temp_Sim)
        for l in range(len(FireLevel_List)):
            if FireLevel_Sum == 0:
                FireLevel_Norm = 0
                mu.append(FireLevel_Norm)
            else:
                FireLevel_Norm = FireLevel_List[l] / FireLevel_Sum
                mu.append(FireLevel_Norm)
        Mu4Test_list.append(mu)
    Mu4Test_list_Result = np.array(Mu4Test_list)
    return ClusterCenters, Mu4Train_list_Result, Mu4Val_list_Result, Mu4Test_list_Result
