
import numpy as np
from grakel import PropagationAttr
from grakel.datasets import fetch_dataset
from sklearn.model_selection import train_test_split


def GetSim(G):
    gk = PropagationAttr(normalize=True)
    K = gk.fit_transform(G)
    return K[1][0]


def GetAnt(G_train, G_test, Rules):
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
            exec('Sim{} = GetSim(CompareGraph{})'.format(k, k))

        for l in range(Rules):
            exec('Sims.append(Sim{})'.format(l))

        maxsim = max(Sims)
        label = Sims.index(max(Sims))
        exec('Cluser{}.append(g)'.format(label))

    for K in range(Rules):
        exec(
            'SumSimCluster{} = []\nfor i in range(len(Cluser{})):\n    SumSim = 0\nfor j in range(len(Cluser{})):\n    temp_compare_graphs = [Cluser{}[i], Cluser{}[j]]\ntemp_sim = GetSim(temp_compare_graphs)\nSumSim += temp_sim\nSumSimCluster{}.append(SumSim)'.format(
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


def Cluster(DatasetName, Rules):
    # DatasetName:"ENZYMES"
    Dataset = fetch_dataset(DatasetName, prefer_attr_nodes=True, verbose=False)
    G, y = Dataset.data, Dataset.target

    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=0)

    # # Define number of rules in GFS
    # Rules = 5

    # Step-1 Initialize the cluster prototype graph
    InitProtGraphIds = np.random.randint(low=0, high=len(G_train), size=Rules)
    PrototypeGraphs = []
    # ClusterTemp = []
    for id, val in enumerate(InitProtGraphIds):
        PrototypeGraphs.append(G_train[val])

    for i in range(Rules):
        exec('Cluser{} = {}'.format(i, []))

    # Cluser0 = []
    # Cluser1 = []
    # Cluser2 = []
    # Cluser3 = []
    # Cluser4 = []

    # print(PrototypeGraphs)
    for i, g in enumerate(G_train):
        for j in range(Rules):
            exec('CompareGraph{} = {}'.format(j, [PrototypeGraphs[j], g]))
        # CompareGraph0 = [PrototypeGraphs[0], g]
        #
        # CompareGraph1 = [PrototypeGraphs[1], g]
        #
        # CompareGraph2 = [PrototypeGraphs[2], g]
        #
        # CompareGraph3 = [PrototypeGraphs[3], g]
        #
        # CompareGraph4 = [PrototypeGraphs[4], g]

        Sims = []
        for k in range(Rules):
            exec('Sim{} = GetSim(CompareGraph{})'.format(k, k))
        # Sim0 = GetSim(CompareGraph0)
        # Sim1 = GetSim(CompareGraph1)
        # Sim2 = GetSim(CompareGraph2)
        # Sim3 = GetSim(CompareGraph3)
        # Sim4 = GetSim(CompareGraph4)
        for l in range(Rules):
            exec('Sims.append(Sim{})'.format(l))
        # Sims.append(Sim0)
        # Sims.append(Sim1)
        # Sims.append(Sim2)
        # Sims.append(Sim3)
        # Sims.append(Sim4)
        maxsim = max(Sims)
        label = Sims.index(max(Sims))
        exec('Cluser{}.append(g)'.format(label))
        # if label == 0:
        #
        #     Cluser0.append(g)
        # if label == 1:
        #     Cluser1.append(g)
        # if label == 2:
        #     Cluser2.append(g)
        # if label == 3:
        #     Cluser3.append(g)
        # if label == 4:
        #     Cluser4.append(g)
        # print('最大相似度：', maxsim)
        # print('标签：', label)

    # print(len(Cluser0))
    # print(len(Cluser1))
    # print(len(Cluser2))
    # print(len(Cluser3))
    # print(len(Cluser4))

    for K in range(Rules):
        exec(
            'SumSimCluster{} = []\nfor i in range(len(Cluser{})):\n    SumSim = 0\nfor j in range(len(Cluser{})):\n    temp_compare_graphs = [Cluser{}[i], Cluser{}[j]]\ntemp_sim = GetSim(temp_compare_graphs)\nSumSim += temp_sim\nSumSimCluster{}.append(SumSim)'.format(
                K, K, K, K, K, K))


    for K in range(Rules):
        exec(
            'MaxIndex_Cluster{} = SumSimCluster{}.index(max(SumSimCluster{}))\nMaxOfSumSimCluster{} = max(SumSimCluster{})'.format(
                K, K, K, K, K))
    ClusterCenters = []
    for K in range(Rules):
        exec('ClusterCenters.append(Cluser{}[MaxIndex_Cluster{}])'.format(K, K))
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

    Mu4Train_list = np.array(Mu4Train_list)
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

    Mu4Test_list = np.array(Mu4Test_list)
    return ClusterCenters, Mu4Train_list, Mu4Test_list


