
import numpy as np
from grakel import PropagationAttr
from grakel.datasets import fetch_dataset
from sklearn.model_selection import train_test_split


def GetSim(G):
    # Get the graph kernel matrix
    gk = PropagationAttr(normalize=True)
    K = gk.fit_transform(G)
    return K[1][0]


def Cluster(DatasetName, Rules):

    Dataset = fetch_dataset(DatasetName, prefer_attr_nodes=True, verbose=False)
    G, y = Dataset.data, Dataset.target

    G_train, G_test, y_train, y_test = train_test_split(G, y, test_size=0.2, random_state=0)

    InitProtGraphIds = np.random.randint(low=0, high=len(G_train), size=Rules)
    PrototypeGraphs = []
    # ClusterTemp = []
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
    return Mu4Train_list, Mu4Test_list


