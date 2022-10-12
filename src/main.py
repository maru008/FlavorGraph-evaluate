import os
import pandas as pd
from tabulate import tabulate

from units import MaltiClassification_unit

# enbd_data_ls = os.listdir("../embedding_data")
enbd_data_ls = [
                # 'Metapath2vec.pickle',
                # 'Metapath2vec+NPMI.pickle',
                'Metapath2vec+CSP.pickle',
                'Metapath2vec+NPMI+CSP.pickle',
                # 'Metapath2vec+HiddenFP.pickle',
                # 'Metapath2vec+Hidden_PubchemFP_100.pickle',
                # 'Metapath2vec+Hidden_PubchemFP_300.pickle',
                # 'Metapath2vec+Morgan2binary_445.pickle',
                # 'Metapath2vec+Morgan2binary_300.pickle',
                # 'Metapath2vec+Morgan2binary_100.pickle',
                # 'Metapath2vec+PubchemFP_W2V.pickle',
                # 'Metapath2vec+M2Vtrain_100.pickle',
                # 'Metapath2vec+Morgan2binary_445.pickle',
                # 'Metapath2vec+M2Vtrain_300_w5.pickle',
                # 'Metapath2vec+NPMI+Morgan2binary_445.pickle',
                # 'Metapath2vec+NPMI+Mol2vec_w5.pickle'
                # 'Metapath2vec+M2Vtrain.pickle',
                # 'Metapath2vec+M2Vtrain_300_w15.pickle',
                # 'Metapath2vec+M2Vtrain_300_w20.pickle'
                'Metapath2vec+M2V_chembl_300_w10.pickle',
                'Metapath2vec+NPMI+M2V_chembl_300_w10.pickle'
                ]

def evaluate_main(PATH,train_raito):
    print("===========================================================================================================")
    print("++++++++++++++++++++++++++++++++++++++++++++++++evaluate:{}++++++++++++++++++++++++++++++++++++++++++++++++".format(PATH.replace("../classification_data/","")))
    print("===========================================================================================================")
    MLRclassification_res_matrix = []
    SVMclassification_res_matrix = []
    MLPclassification_res_matrix = []
    XGBoostclassification_res_matrix = []
    LightGBMclassification_res_matrix = []
    CatBoostclassification_res_matrix = []
    for file_i in enbd_data_ls:
        EMBED_PATH = "../embedding_data/" + file_i
        
        Evaluate_Unit = MaltiClassification_unit(PATH,EMBED_PATH,train_raito)
        
        MLRclassification_res_matrix.append(Evaluate_Unit.MaltiLogisticRegresson())
        SVMclassification_res_matrix.append(Evaluate_Unit.SupportVectorMachine())
        MLPclassification_res_matrix.append(Evaluate_Unit.MultilayerPerceptron())
        # XGBoostclassification_res_matrix.append(Evaluate_Unit.XGBoost())
        # LightGBMclassification_res_matrix.append(Evaluate_Unit.LightGBM())
        # CatBoostclassification_res_matrix.append(Evaluate_Unit.CatBoost())
    headers =  ["Method","F1-macro","F1-micro","F1-weighted"]
    print("Multi Logistic Regression")
    result=tabulate(MLRclassification_res_matrix, headers, tablefmt="grid")
    print(result)
    print("SVM")
    result=tabulate(SVMclassification_res_matrix, headers, tablefmt="grid")
    print(result)
    print("MLP")
    result=tabulate(MLPclassification_res_matrix, headers, tablefmt="grid")
    print(result)
    # print("XGBoost")
    # result=tabulate(XGBoostclassification_res_matrix, headers, tablefmt="grid")
    # print(result)
    # print("LigheGBM")
    # result=tabulate(LightGBMclassification_res_matrix, headers, tablefmt="grid")
    # print(result)
    # print("CatBoost")
    # result=tabulate(CatBoostclassification_res_matrix, headers, tablefmt="grid")
    # print(result)
    
PATH = '../classification_data/node_classification_160.csv'
evaluate_main(PATH,0.8)

PATH = '../classification_data/node_classification_416_hub.csv'
evaluate_main(PATH,0.8)

PATH = '../classification_data/node_classification_616.csv'
evaluate_main(PATH,0.8)

# print("=========================================================================================================== train raito 0.75 ===========================================================================================================")
# PATH = '../classification_data/node_classification_160.csv'
# evaluate_main(PATH,0.75)

# PATH = '../classification_data/node_classification_416_hub.csv'
# evaluate_main(PATH,0.75)

# PATH = '../classification_data/node_classification_616.csv'
# evaluate_main(PATH,0.75)

# print("=========================================================================================================== train raito 0.7 ===========================================================================================================")
# PATH = '../classification_data/node_classification_160.csv'
# evaluate_main(PATH,0.7)

# PATH = '../classification_data/node_classification_416_hub.csv'
# evaluate_main(PATH,0.7)

# PATH = '../classification_data/node_classification_616.csv'
# evaluate_main(PATH,0.7)




    
    




