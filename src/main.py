from email import header
from operator import index
from unittest import result
import evaluate_unit
import os
import pandas as pd
from tabulate import tabulate

enbd_data_ls = os.listdir("../embedding_data")
# enbd_data_ls = ["Metapath2vec+CSP_8212.pickle","Metapath2vec+M2V_weighted.pickle","Metapath2vec+NPMI+M2V_weighted.pickle"]
enbd_data_ls = ['Metapath2vec+CSP.pickle',
                'Metapath2vec+M2Vtrain.pickle','Metapath2vec+HiddenFP.pickle','Metapath2vec+PubchemFP_W2V.pickle','Metapath2vec+Morgan2binary.pickle',
                'Metapath2vec+NPMI+CSP.pickle','Metapath2vec+NPMI+M2Vtrain.pickle','Metapath2vec+NPMI+node2fp_W2V.pickle','Metapath2vec+NPMI+Morgan2binary.pickle']
# enbd_data_ls = ['Metapath2vec+FoodM2V.pickle']
# 'Metapath2vec+M2V.pickle','Metapath2vec+NPMI+M2V.pickle','Metapath2vec+M2V_partweighted_MinMax.pickle','Metapath2vec+NPMI+M2V_partweighted_MinMax.pickle','Metapath2vec+M2V_partweighted_TFIDF.pickle',,'Metapath2vec+NPMI+M2V_partweighted_IFIDF.pickle'
def evaluate_main(data_len):
    
    print("===========================================================================================================")
    print("++++++++++++++++++++++++++++++++++++++++++++++++evaluate:{}++++++++++++++++++++++++++++++++++++++++++++++++".format(data_len))
    print("===========================================================================================================")
    MLRclassification_res_matrix = []
    SVMclassification_res_matrix = []
    MLPclassification_res_matrix = []
    KMEANSclassification_res_matrix = []
    for file_i in enbd_data_ls:
        print(file_i)
        file_i = "../embedding_data/" + file_i
        res_ls = evaluate_unit.evaluate(file_i,"../classification_data/node_classification_{}.csv".format(data_len))
        # print(file_i)
        MLRclassification_res_matrix.append(res_ls[0])
        SVMclassification_res_matrix.append(res_ls[1])
        MLPclassification_res_matrix.append(res_ls[2])
        KMEANSclassification_res_matrix.append(res_ls[3])
    print("Multi Logistic Regression")
    headers =  ["Method","MLR Accuracy","MLR F1-macro","MLR F1-micro"]
    result=tabulate(MLRclassification_res_matrix, headers, tablefmt="grid")
    print(result)
    print("SVM")
    headers= ["Method","SVM Accuracy","SVM F1-macro","SVM F1-micro"]
    result=tabulate(SVMclassification_res_matrix, headers, tablefmt="grid")
    print(result)
    print("MLP")
    headers= ["Method","MLP Accuracy","MLP F1-macro","MLP F1-micro"]
    result=tabulate(MLPclassification_res_matrix, headers, tablefmt="grid")
    print(result)
    print("Kmeans ++")
    headers =  ["Method","Kmeans++ NMI mean","Kmeans++ NMI std"]
    result=tabulate(KMEANSclassification_res_matrix, headers, tablefmt="grid")
    print(result)
    
    
evaluate_main(160)
evaluate_main(616)

    
    




