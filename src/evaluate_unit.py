import pandas as pd
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier


# class evaluate_classification:
#     def __init__(self,file_name):
        
    

def node_name2vec(name,file):
    with open(file, "rb") as pickle_file:
        vectors = pickle.load(pickle_file)
    node_data = pd.read_csv("../input/nodes_8212.csv")
    name = name.replace(' ',"_")
    id = str(node_data[node_data["name"] == name]["node_id"].values[0])
    vec = vectors[id]
    return vec

def evaluate(file,data_path):
    df = pd.read_csv(data_path)
    # categories = list(set(df["category"]))
    method_name = file.replace(".pickle","").replace("../embedding_data/","")
    # with open(file, "rb") as pickle_file:
    #     vectors = pickle.load(pickle_file)
    X = []
    y = []
    for name,categ in zip(df["ingredient"],df["category"]):
        try:
            vec_i = node_name2vec(name,file)
            X.append(vec_i)
            y.append(categ)
        except:
            pass
        
    # for category in categories:
    #     ingredients = df[category].values
    #     for name in ingredients:
    #         vec = node_name2vec(name,file)
    #         X.append(vec)
    #         y.append(category)
    
    MLR_res = [method_name]
    SVM_res = [method_name]
    MLP_res = [method_name]
    KMEAS_res = [method_name]
    
    MLRclassification_res_ls,SVMclassification_res_ls,MLPclassification_res_ls = classification_train(X, y, 0.8)
    MLR_res.extend(MLRclassification_res_ls)
    SVM_res.extend(SVMclassification_res_ls)
    MLP_res.extend(MLPclassification_res_ls)
    
    train_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    nmis = []
    for ratio in train_ratios:
        nmi = NMI_train(X, y, ratio)
        nmis.append(nmi)
    KMEAS_res.append(np.mean(nmis))
    KMEAS_res.append(np.std(nmis))
    res_ls = [MLR_res,SVM_res,MLP_res,KMEAS_res]
    return res_ls

def NMI_train(X, y, train_ratio):
    """
    Clustering
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics.cluster import normalized_mutual_info_score
    from nltk.cluster import KMeansClusterer
    import nltk
    NUM_CLUSTERS= len(set(y))
    # kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=100, normalise=True, avoid_empty_clusters=True)
    # assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
    # nmi = normalized_mutual_info_score(assigned_clusters, y)
    
    res_clusters = KMeans(n_clusters=NUM_CLUSTERS, init='k-means++',random_state= int(train_ratio*10)).fit_predict(X)
    nmi = normalized_mutual_info_score(res_clusters, y)
    return nmi


def classification_train(X, y, train_ratio):
    from sklearn.linear_model import LogisticRegression
    from sklearn import svm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score, f1_score
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    
    # csv = "input/node_classification_hub.csv"
    # df = pd.read_csv(csv)
    # categories = df.columns
    
    test_ratio = 1-train_ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=36)
    MLRclassification_res_ls = []
    SVMclassification_res_ls = []
    MLPclassification_res_ls = []
    """
    Classification
    """
    clf = LogisticRegression(C=1000.0, random_state=0,max_iter=1500).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print("accuracy: %f" %accuracy_score(y_test, y_pred))
    # print("Precison")
    # precison_ls = precision_score(y_test, y_pred,average = None,labels = categories)
    # for l_i,p_i in zip(categories,precison_ls):
    #     print(l_i ," : %f" % p_i)
    # print("Recall")
    # recall_ls = recall_score(y_test, y_pred,average = None,labels = categories)
    # for l_i,r_i in zip(categories,recall_ls):
    #     print(l_i ," : %f" % r_i)
    
    # print("F1-macro : %f" % f1_score(y_test, y_pred, average='macro'))
    # print("F1-micro : %f" % f1_score(y_test, y_pred, average='micro'))
    
    MLRclassification_res_ls.append(accuracy_score(y_test, y_pred))
    MLRclassification_res_ls.append(f1_score(y_test, y_pred, average='macro'))
    MLRclassification_res_ls.append(f1_score(y_test, y_pred, average='micro'))

    # print("-------- SVM --------")
    clf = svm.SVC(kernel='linear', C=1e30).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print("accuracy: %f" %accuracy_score(y_test, y_pred))
    # print("Precision : %.3f" % precision_score(y_test, y_pred,average = 'None'))
    # print("Recall : %.3f" % recall_score(y_test, y_pred,average = 'None'))
    # print("F1-macro : %f" % f1_score(y_test, y_pred, average='macro'))
    # print("F1-weight : %f" % f1_score(y_test, y_pred, average = 'weighted'))
    # print("F1-micro : %f" % f1_score(y_test, y_pred, average='micro'))
    SVMclassification_res_ls.append(accuracy_score(y_test, y_pred))
    SVMclassification_res_ls.append(f1_score(y_test, y_pred, average='macro'))
    SVMclassification_res_ls.append(f1_score(y_test, y_pred, average='micro'))
    
    
    # print("-------- MLP --------")
    clf =  MLPClassifier(max_iter=10000,random_state = 0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # print("accuracy: %f" %accuracy_score(y_test, y_pred))
    # print("F1-macro : %f" % f1_score(y_test, y_pred, average='macro'))
    # print("F1-weight : %f" % f1_score(y_test, y_pred, average = 'weighted'))
    # print("F1-micro : %f" % f1_score(y_test, y_pred, average='micro'))
    
    MLPclassification_res_ls.append(accuracy_score(y_test, y_pred))
    MLPclassification_res_ls.append(f1_score(y_test, y_pred, average='macro'))
    MLPclassification_res_ls.append(f1_score(y_test, y_pred, average='micro'))
    return MLRclassification_res_ls,SVMclassification_res_ls,MLPclassification_res_ls
        
    
