scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/my_IPSJ2022_CSP_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/my_IPSJ2022_CSP.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data

+NPMI+CSP
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI/my_output/my_IPSJ2022_CSP_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_my_IPSJ2022/my_output/my_IPSJ2022.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/input/node_classification_hub.csv /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/input
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_False-isCSP_0.0001-CSPcoef.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data

化学構造予測層の追加
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_my_CEA2022++/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_my_CEA2022++/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data 

Mol2vec+NPMI
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI+M2V/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data 

以下少ないノード
Metapath2vec only 
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src8298/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_False-isCSP_0.0001-CSPcoef.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
Metapath2vec＋CSP only 
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src8298/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data

Metapath2vec＋NPMI
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI_8298/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_False-isCSP_0.0001-CSPcoef.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
Metapath2vec＋NPMI+CSP
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI_8298/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data

重み付き
minmax
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V_weighted/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI+M2V_weighted/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
zsocre
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V_weighted/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI+M2V_weighted/output/FlavorGraph+CSL-embedding_M11-metapath_300-dim_0.0025-initial_lr_3-window_size_10-iterations_5-min_count-_True-isCSP_0.0001-CSPcoef_CSPLayer.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data

TDIDF
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V_weighted/output/Metapath2vec+M2V_partweighted_TFIDF.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI+M2V_weighted/output/Metapath2vec+NPMI+M2V_partweighted_TFIDF.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data

FoodMol2vec
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V/output/Metapath2vec+FoodM2V.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/src_my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI+M2V_weighted/output/Metapath2vec+NPMI+FoodM2V.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/my_evaluat/embedding_data

scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V/output/Metapath2vec+FoodM2V_weighted.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI+M2V_weighted/output/Metapath2vec+NPMI+FoodM2V_weighted.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/my_evaluat/embedding_data

scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V/output/Metapath2vec+M2Vtrain.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/my_evaluat/embedding_data
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+NPMI+M2V/output/Metapath2vec+NPMI+M2Vtrain.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/my_evaluat/embedding_data

考察
scp miluser@192.168.100.67:/home/miluser/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V/output/Metapath2vec+hiddenFP.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/my_evaluat/embedding_data

scp miluser@192.168.100.66:/home/miluser/yoshimaru_citrine_backup/yoshimaru_env/FlavorGraph/src_Metapath2vec+M2V/output/Metapath2vec+PubchemFP_W2V.pickle /Users/naokiyoshimaru/Desktop/pro-FlavorGraph_evaluate/my_evaluat/embedding_data