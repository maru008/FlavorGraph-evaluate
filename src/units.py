import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
import lightgbm as lgb
import optuna.integration.lightgbm as opt_lgb
from catboost import Pool,CatBoost
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class MaltiClassification_unit:
    def __init__(self,csv_path,embed_path,ratio:float):
        eval_data = pd.read_csv(csv_path)
        X = []
        y = []
        # print("Loading Embedding Data({})...".format(embed_path.replace("../embedding_data/","")),end="")
        for ctg , name in zip(eval_data["category"],eval_data["ingredient"]):
            vec = node_name2vec(name,embed_path)
            if vec is not None:
                X.append(vec)
                y.append(ctg)
        test_ratio = 1 - ratio
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=36)
        
        self.X = X
        self.y = y
        self.train_ratio = ratio
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.label_ls = list(set(y))
        self.classNum = len(list(set(y)))
        self.method = embed_path.replace("../embedding_data/","").replace(".pickle","")
        
        
    def MaltiLogisticRegresson(self):
        """
        Running LogisticRegresson functuon.
        
        This functuon can evaluate multi class.
        """
        # X_train, X_test, y_train, y_test = self.create_train_data()
        clf = LogisticRegression(C=1000.0, random_state=0,max_iter=1500).fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        
        macroF1_value = f1_score(self.y_test, y_pred, average='macro')
        microF1_value = f1_score(self.y_test, y_pred, average='micro')
        weightF1_value = f1_score(self.y_test, y_pred, average='weighted')
        return [self.method,macroF1_value,microF1_value,weightF1_value]
        
    def SupportVectorMachine(self):
        clf = svm.SVC(kernel='linear', C=1e30).fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        
        macroF1_value = f1_score(self.y_test, y_pred, average='macro')
        microF1_value = f1_score(self.y_test, y_pred, average='micro')
        weightF1_value = f1_score(self.y_test, y_pred, average='weighted')
        
        return [self.method,macroF1_value,microF1_value,weightF1_value]
    def RandomForest(self):
        clf = RandomForestClassifier(n_estimators = self.classNum, random_state = 0)
        clf.fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)

        macroF1_value = f1_score(self.y_test, y_pred, average='macro')
        microF1_value = f1_score(self.y_test, y_pred, average='micro')
        weightF1_value = f1_score(self.y_test, y_pred, average='weighted')
        
        return [self.method,macroF1_value,microF1_value,weightF1_value]
        
    def MultilayerPerceptron(self):
        clf =  MLPClassifier(max_iter=10000,random_state = 0).fit(self.X_train, self.y_train)
        y_pred = clf.predict(self.X_test)
        
        macroF1_value = f1_score(self.y_test, y_pred, average='macro')
        microF1_value = f1_score(self.y_test, y_pred, average='micro')
        weightF1_value = f1_score(self.y_test, y_pred, average='weighted')
        
        return [self.method,macroF1_value,microF1_value,weightF1_value]
        
    def XGBoost(self):
        # ?????????????????????????????????
        LE = LabelEncoder()
        y_label =  LE.fit_transform(self.y)
        num_class = len(set(y_label)) #Food category class number
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, y_label, test_size=(1-self.train_ratio), random_state=36)
        # X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,test_size=0.2,random_state=2,stratify=y_train)
        
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest  = xgb.DMatrix(X_test, label=y_test)
        params = {'max_depth': 3, 'eta': 1, 'objective': 'multi:softprob', 'num_class': num_class}
        
        num_round = 100
        watchlist = [(dtrain, 'train'), (dtest, 'test')]
        model = xgb.train(params, dtrain, num_round, verbose_eval=0, evals=watchlist)
        y_pred = model.predict(dtest).argmax(axis=1)
        
        macroF1_value = f1_score(y_test, y_pred, average='macro')
        microF1_value = f1_score(y_test, y_pred, average='micro')
        weightF1_value = f1_score(y_test, y_pred, average='weighted')
        
        return [self.method,macroF1_value,microF1_value,weightF1_value]
    
    def LightGBM(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=(1-self.train_ratio), random_state=36)
        lgb_train = opt_lgb.Dataset(X_train, y_train)
        lgb_test = opt_lgb.Dataset(X_test, y_test, reference=lgb_train)
        
        print(len(X_train))
        print("*")
        
        params = {
            'task': 'train',  #?????????????????????
            'boosting_type': 'gbdt', #????????????????????????????????????
            'objective': 'multiclass', #?????????????????????
            'num_class': self.classNum , #??????????????????
            'metric': 'multi_logloss' #??????????????????????????????Log??????
        }
        print("*")
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=lgb_test,
            num_boost_round=100,
            early_stopping_rounds=self.classNum
        )
        
        print("*")
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)
        
        macroF1_value = f1_score(y_test, y_pred, average='macro')
        microF1_value = f1_score(y_test, y_pred, average='micro')
        weightF1_value = f1_score(y_test, y_pred, average='weighted')
        
        return [self.method,macroF1_value,microF1_value,weightF1_value]
    
    def CatBoost(self):
        train_pool = Pool(self.X_train, self.y_train)
        test_pool = Pool(self.X_test, self.y_test)
        params = {
            'loss_function': 'Logloss',
            # ?????????????????????
            'num_boost_round': 100
        }
        
        model = CatBoostClassifier(iterations=2, learning_rate=1, depth=2, loss_function='Logloss')
        model.fit(train_pool, cat_features = self.label_ls)
        
        y_pred = model.predict(test_pool, prediction_type='Class')
        
        macroF1_value = f1_score(self.y_test, y_pred, average='macro')
        microF1_value = f1_score(self.y_test, y_pred, average='micro')
        weightF1_value = f1_score(self.y_test, y_pred, average='weighted')
        
        return [self.method,macroF1_value,microF1_value,weightF1_value]

def node_name2vec(name,file):
    """
    This is function for convert nodeName to vector.
    
    Input argument type is string and output data is float vector list.
    
    1:node name. 2:file name.
    """
    with open(file, "rb") as pickle_file:
        vectors = pickle.load(pickle_file)
    node_data = pd.read_csv("../input/nodes_8212.csv")
    node_name_ls = list(node_data["name"].values)
    name = name.replace(' ',"_")
    if name in node_name_ls:
        id = str(node_data[node_data["name"] == name]["node_id"].values[0])
        vec = vectors[id]
    else:
        vec = None
    return vec

        
#??????????????????????????????
class embedding_plot:
    def __init__(self,filePath):
        self.em_file_Path = f'../embedding_data/{filePath}'
        print(f'../embedding_data/{filePath}')
        with open(self.em_file_Path, 'rb') as f:
            trg_pickledata = pickle.load(f)
        self.trg_pickledata = trg_pickledata
        
    def plot_tSNE(self):
        categories = pd.read_csv("../classification_data/node_classification_hub.csv").columns
        classification_data = pd.read_csv("../classification_data/node_classification_160.csv")
        trg_df = pd.DataFrame()
        for category in categories:
            add_df = classification_data[classification_data["category"] == category].sample(random_state=0)
            trg_df = pd.concat([trg_df,add_df])
        print(type(self.em_file_Path))
        tsne = TSNE(n_components=2, random_state = 0, n_iter = 1000)
        vec_ls = [node_name2vec(name,self.em_file_Path) for name in trg_df["ingredient"]]
        embed_vec = tsne.fit_transform(vec_ls)
        trg_df["embed_1"] = embed_vec.T[0]
        trg_df["embed_2"] = embed_vec.T[1]
        plt.figure(figsize=(8,6))
        colors =  ["r", "g", "c", "m", "y", "k", "orange","pink"]
        for i , cat in enumerate(categories):
            print(i)
            tmp_df = trg_df[trg_df["category"] == cat]
            plt.scatter(tmp_df["embed_1"], tmp_df["embed_2"],label = cat,color = colors[i])
        plt.legend(fontsize = 8)
        plt.show()
        


    