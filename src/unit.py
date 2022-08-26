import pickle
import pandas as pd

def node_name2vec(name,file):
    with open(file, "rb") as pickle_file:
        vectors = pickle.load(pickle_file)
    node_data = pd.read_csv("../input/nodes_191120.csv")
    id = str(node_data[node_data["name"] == name]["node_id"].values[0])
    vec = vectors[id]
    return vec