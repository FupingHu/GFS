import os
import numpy as np
from grakel.graph import Graph
from collections import Counter
from sklearn.utils import Bunch

dataset_metadata = {
    "freesolv": {"nl": True, "el": False, "na": True, "ea": False},
    "esol": {"nl": True, "el": False, "na": True, "ea": False},
    "herg": {"nl": True, "el": False, "na": True, "ea": False},
    "Lipophilicity": {"nl": True, "el": False, "na": True, "ea": False},
    "qm9": {"nl": True, "el": False, "na": True, "ea": False},
    "aspirin": {"nl": True, "el": False, "na": True, "ea": False},
    "AIDS": {"nl": True, "el": True, "na": True, "ea": False},
    "BZR": {"nl": True, "el": False, "na": True, "ea": False},
    "BZR_MD": {"nl": True, "el": True, "na": False, "ea": True},
    "COIL-DEL": {"nl": False, "el": True, "na": True, "ea": False},
    "COIL-RAG": {"nl": False, "el": False, "na": True, "ea": True},
    "COLLAB": {"nl": False, "el": False, "na": False, "ea": False},
    "COX2": {"nl": True, "el": False, "na": True, "ea": False},
    "COX2_MD": {"nl": True, "el": True, "na": False, "ea": True},
    "DHFR": {"nl": True, "el": False, "na": True, "ea": False},
    "DHFR_MD": {"nl": True, "el": True, "na": False, "ea": True},
    "ER_MD": {"nl": True, "el": True, "na": False, "ea": True},
    "DD": {"nl": True, "el": False, "na": False, "ea": False},
    "ENZYMES": {"nl": True, "el": False, "na": True, "ea": False},
    "Cuneiform": {"nl": True, "el": True, "na": True, "ea": True},
    "FINGERPRINT": {"nl": False, "el": False, "na": True, "ea": True},
    "FIRSTMM_DB": {"nl": True, "el": False, "na": True, "ea": True},
    "FRANKENSTEIN": {"nl": False, "el": False, "na": True, "ea": False},
    "IMDB-BINARY": {"nl": False, "el": False, "na": False, "ea": False},
    "IMDB-MULTI": {"nl": False, "el": False, "na": False, "ea": False},
    "Letter-high": {"nl": False, "el": False, "na": True, "ea": False},
    "Letter-low": {"nl": False, "el": False, "na": True, "ea": False},
    "Letter-med": {"nl": False, "el": False, "na": True, "ea": False},
    "Mutagenicity": {"nl": True, "el": True, "na": False, "ea": False},
    "MSRC_9": {"nl": True, "el": False, "na": False, "ea": False},
    "MSRC_21": {"nl": True, "el": False, "na": False, "ea": False},
    "MSRC_21C": {"nl": True, "el": False, "na": False, "ea": False},
    "MUTAG": {"nl": True, "el": True, "na": False, "ea": False},
    "NCI1": {"nl": True, "el": False, "na": False, "ea": False},
    "NCI109": {"nl": True, "el": False, "na": False, "ea": False},
    "PTC_FM": {"nl": True, "el": True, "na": False, "ea": False},
    "PTC_FR": {"nl": True, "el": True, "na": False, "ea": False},
    "PTC_MM": {"nl": True, "el": True, "na": False, "ea": False},
    "PTC_MR": {"nl": True, "el": True, "na": False, "ea": False},
    "PROTEINS": {"nl": True, "el": False, "na": True, "ea": False},
    "PROTEINS_full": {"nl": True, "el": False, "na": True, "ea": False},
    "REDDIT-BINARY": {"nl": False, "el": False, "na": False, "ea": False},
    "REDDIT-MULTI-5K": {"nl": False, "el": False, "na": False, "ea": False},
    "REDDIT-MULTI-12K": {"nl": False, "el": False, "na": False, "ea": False},
    "SYNTHETIC": {"nl": False, "el": False, "na": True, "ea": False},
    "SYNTHETICnew": {"nl": False, "el": False, "na": True, "ea": False},
    "Synthie": {"nl": False, "el": False, "na": True, "ea": False},
    "Tox21_AHR": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_AR": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_AR-LBD": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_ARE": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_aromatase": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_ATAD5": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_ER": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_ER_LBD": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_HSE": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_MMP": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_p53": {"nl": True, "el": True, "na": False, "ea": False},
    "Tox21_PPAR-gamma": {"nl": True, "el": True, "na": False, "ea": False}
}

symmetric_dataset = False

def read_regression_data(
        name,
        with_classes=True,
        prefer_attr_nodes=False,
        prefer_attr_edges=False,
        produce_labels_nodes=False,
        as_graphs=False,
        is_symmetric=symmetric_dataset):
    """Create a dataset iterable for GraphKernel.

    Parameters
    ----------
    name : str
        The dataset name.

    with_classes : bool, default=False
        Return an iterable of class labels based on the enumeration.

    produce_labels_nodes : bool, default=False
        Produce labels for nodes if not found.
        Currently this means labeling its node by its degree inside the Graph.
        This operation is applied only if node labels are non existent.

    prefer_attr_nodes : bool, default=False
        If a dataset has both *node* labels and *node* attributes
        set as labels for the graph object for *nodes* the attributes.

    prefer_attr_edges : bool, default=False
        If a dataset has both *edge* labels and *edge* attributes
        set as labels for the graph object for *edge* the attributes.

    as_graphs : bool, default=False
        Return data as a list of Graph Objects.

    is_symmetric : bool, default=False
        Defines if the graph data describe a symmetric graph.

    Returns
    -------
    Gs : iterable
        An iterable of graphs consisting of a dictionary, node
        labels and edge labels for each graph.

    classes : np.array, case_of_appearance=with_classes==True
        An one dimensional array of graph classes aligned with the lines
        of the `Gs` iterable. Useful for classification.

    """
    # root_path = "./datasets/"
    indicator_path = "./datasets/" + str(name) + "/" + str(name) + "_graph_indicator.txt"
    edges_path = "./datasets/" + str(name) + "/" + str(name) + "_A.txt"
    node_labels_path = "./datasets/" + str(name) + "/" + str(name) + "_node_labels.txt"
    node_attributes_path = "./datasets/" + str(name) + "/" + str(name) + "_node_attributes.txt"
    edge_labels_path = "./datasets/" + str(name) + "/" + str(name) + "_edge_labels.txt"
    edge_attributes_path = \
        "./datasets/" + str(name) + "/" + str(name) + "_edge_attributes.txt"
    graph_classes_path = \
        "./datasets/" + str(name) + "/" + str(name) + "_graph_attributes.txt"

    # node graph correspondence
    ngc = dict()
    # edge line correspondence
    elc = dict()
    # dictionary that keeps sets of edges
    Graphs = dict()
    # dictionary of labels for nodes
    node_labels = dict()
    # dictionary of labels for edges
    edge_labels = dict()

    # Associate graphs nodes with indexes
    with open(indicator_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            ngc[i] = int(line[:-1])
            if int(line[:-1]) not in Graphs:
                Graphs[int(line[:-1])] = set()
            if int(line[:-1]) not in node_labels:
                node_labels[int(line[:-1])] = dict()
            if int(line[:-1]) not in edge_labels:
                edge_labels[int(line[:-1])] = dict()

    # Extract graph edges
    with open(edges_path, "r") as f:
        for (i, line) in enumerate(f, 1):
            edge = line[:-1].replace(' ', '').split(",")
            elc[i] = (int(edge[0]), int(edge[1]))
            Graphs[ngc[int(edge[0])]].add((int(edge[0]), int(edge[1])))
            if is_symmetric:
                Graphs[ngc[int(edge[1])]].add((int(edge[1]), int(edge[0])))

    # Extract node attributes
    if (prefer_attr_nodes and
            dataset_metadata[name].get(
                "na",
                os.path.exists(node_attributes_path)
            )):
        with open(node_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = \
                    [float(num) for num in
                     line[:-1].replace(' ', '').split(",")]
    # Extract node labels
    elif dataset_metadata[name].get(
            "nl",
            os.path.exists(node_labels_path)
    ):
        with open(node_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                node_labels[ngc[i]][i] = int(line[:-1])
    elif produce_labels_nodes:
        for i in range(1, len(Graphs) + 1):
            node_labels[i] = dict(Counter(s for (s, d) in Graphs[i] if s != d))

    # Extract edge attributes
    if (prefer_attr_edges and
            dataset_metadata[name].get(
                "ea",
                os.path.exists(edge_attributes_path)
            )):
        with open(edge_attributes_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                attrs = [float(num)
                         for num in line[:-1].replace(' ', '').split(",")]
                edge_labels[ngc[elc[i][0]]][elc[i]] = attrs
                if is_symmetric:
                    edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = attrs

    # Extract edge labels
    elif dataset_metadata[name].get(
            "el",
            os.path.exists(edge_labels_path)
    ):
        with open(edge_labels_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge_labels[ngc[elc[i][0]]][elc[i]] = int(line[:-1])
                if is_symmetric:
                    edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = \
                        int(line[:-1])

    Gs = list()
    if as_graphs:
        for i in range(1, len(Graphs) + 1):
            Gs.append(Graph(Graphs[i], node_labels[i], edge_labels[i]))
    else:
        for i in range(1, len(Graphs) + 1):
            Gs.append([Graphs[i], node_labels[i], edge_labels[i]])

    if with_classes:
        classes = []
        with open(graph_classes_path, "r") as f:
            for line in f:
                classes.append(float(line[:-1]))

        classes = np.array(classes)
        return Bunch(data=Gs, target=classes)
    else:
        return Bunch(data=Gs)


if __name__ == '__main__':
    # PROTEINS ENZYMES BZR COX2 DHFR PROTEINS_full AIDS aspirin
    DatasetName = 'PROTEINS_full'
    # Data = read_data(DatasetName, prefer_attr_nodes=True)
    # print(Data.data)
    # print(Data.target)
    # print(DatasetName)
