#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
    CLASS for community detection in graphs, both spatial and temporal, functions necessary for the station-fault analysis
'''

class cluster_quality_metric():

    def __init__(self):

        self._l_reg_methods = ['Absolute','Average','minPoints','None']

        self._max_distance=30.0
        self._minimum_samples=3
        self._algorithm=None
        self._metric=None
        self._cluster_method=None
        self._seed=None
#d        self._force_minPts = True     # force to remove subgraphs with < minPts
        self._force_regularity='minPoints' # force Absolute, Average, or None regularity
        self._reg_tol_scaler=0.99     # multiply the regularity by the scaler to reduce the threshold

        pass

    def set_quality_frame(self, clustering_name: str="greedy_modularity_communities",
                          **metric_params: dict):

        import traceback
        import numpy as np

        self._max_distance=None
        self._minimum_samples=None
        self._algorithm=None
        self._metric=None
        self._cluster_method=None
        self._seed=None

        self._name=clustering_name
        try:
            ''' Set the default paramters for the specific clustering method '''
            if 'distance_km' in metric_params:
                if isinstance(metric_params["distance_km"],float) and metric_params["distance_km"] > 0:
                    self._max_distance=metric_params["distance_km"]
                else:
                    raise ValueError('distance_km %s must be a float > 0.'
                                     % str(metric_params["distance_km"]))

            if 'minimum_samples' in metric_params:
                if isinstance(metric_params["minimum_samples"],(int, np.integer)) and metric_params["minimum_samples"] > 0:
#                if metric_params["minimum_samples"] > 0:
                    self._minimum_samples=int(metric_params["minimum_samples"])
                else:
                    raise ValueError('minimum_samples %s must be an int > 0.'
                                     % str(metric_params["minimum_samples"]))

            if 'algorithm' in metric_params:
                if isinstance(metric_params["algorithm"],str):
                    self._algorithm=metric_params["algorithm"]
                else:
                    raise ValueError('algorithm %s is invalid.' % (metric_params["algorithm"]))

            if 'metric' in metric_params:
                if isinstance(metric_params["metric"], str):
                    self._metric=metric_params["metric"]
                else:
                    raise ValueError('metric %s is invalid.' % (metric_params["metric"]))

            if 'cluster_method' in metric_params:
                if isinstance(metric_params["cluster_method"], str):
                    self._cluster_method=metric_params["cluster_method"]
                else:
                    raise ValueError('cluster_method %s is invalid.' % (metric_params["cluster_method"]))

            if 'seed' in metric_params:
                if metric_params["seed"] is np.random:
                    self._seed="random"
                elif metric_params["seed"] is int:
                    self._seed="integer"
                else:
                    self._seed=None

        except Exception as err:
            print("Class cluster_quality_metric [set_quality_frame] Error message:", err)
            print(traceback.format_exc())

        return self


    def get_seq_params(self, _iter_combos_df, _n_exp_seq: int = 0):

#d        import pandas as pd     # holds the clustering sequence parameters
        import numpy as np      # necessary when using seed = np.random
        import traceback

        '''
        Set clustering parameters to execute cloud or graph clustering technique

        TODO: use below dictionaries to look up and validate valuses
        (python dictionary key look up not working)

        _dict_algorithms = {"DBSCAN": ["auto", "ball_tree", "kd_tree", "brute"],
                            'HDBSCAN': ["best","generic","prims_kdtree","prims_balltree","boruvka_kdtree",
                             "boruvka_balltree"]}
        _dict_clust_method = {"HDBSCAN": ["leaf","eom"],"OPTICS": ["xi","dbscan"]}
        _dict_algorithms = {"DBSCAN": ["auto", "ball_tree", "kd_tree", "brute"],
                            "HDBSCAN": "best","generic","prims_kdtree","prims_balltree",
                            "boruvka_kdtree","boruvka_balltree",
                            "OPTICS": ['auto', 'ball_tree', 'kd_tree', 'brute']
                            }

        TODO: acquire the lists and dictionaries from the respective cloud and graph clustering classes
        '''

        _l_cluster_techniques = ['cloud','graph']
        _l_cloud_cluster_name = ['DBSCAN','HDBSCAN','AFFINITYPROPAGATION','OPTICS','MEANSHIFT',
                                  'AGGLOMERATIVE','BIRCH','KMEANS','KNN','DENCLUE',"SPECTRAL"]
        _l_graph_cluster_name = ["GREEDY","NAIVE-GREEDY","LPC","ASYNC-LPA",
                                 "LUKES","ASYNC-FLUID","GIRVAN-NEWMAN"]
        _dict_algorithms = {"auto", "ball_tree", "kd_tree", "brute",
                            "best","generic","prims_kdtree","prims_balltree","boruvka_kdtree","boruvka_balltree",
                            "kmeans","discretize"}

        _dict_clust_method = {"leaf","eom","xi","dbscan","arpack", "lobpcg", "amg"}
        '''In all instances when possible <haversine> will be the choice; else it will be <precomputed>'''
        _lst_metric = ["haversine","euclidean","manhattan","minkowski","precomputed",
                       "precomputed_nearest_neighbors","nearest_neighbors","rbf"]

        _dict_clust_params = {}

        i=_n_exp_seq
        try:
#            ''' Load data from CSV '''
#            _iter_combos_df = pd.read_csv("../experiments/cluster_runs.csv")

            ''' Clustering method name is mandatory'''
            if _iter_combos_df.loc[i, 'name'] not in _l_cloud_cluster_name+_l_graph_cluster_name:
                raise AttributeError('%s is not a valid clustering method use \n%s'
                                     % (_iter_combos_df.loc[i, 'name'],
                                        _l_cloud_cluster_name+_l_graph_cluster_name))
            else:
                _s_cloud_clust_name = str(_iter_combos_df.loc[i, 'name'])

            ''' Technique - assign the appropriate value based on the clustering name '''
            if _iter_combos_df.loc[i, 'technique'] not in _l_cluster_techniques:
                if _s_cloud_clust_name in _l_cloud_cluster_name:
                    _cluster_technique = "cloud"
                elif _s_cloud_clust_name in _l_graph_cluster_name:
                    _cluster_technique = "graph"
            else:
                _cluster_technique = str(_iter_combos_df.loc[i, 'technique'])

            ''' Create clustering input parameter Dictionary
                Maximum distance between points and the minimum points are undefined assign defaults '''
            if _iter_combos_df.loc[i, 'maxDistance'].astype(float) <= 1:
                print('maxDistance (Km) must ba a float >= 1.0; proceeding with default maxDistance=30.0 Km')
                _dict_clust_params["distance_km"] = 30.0
            else:
#                _dict_clust_params["distance_km"] = _iter_combos_df.loc[i, 'maxDistance'].astype(float)
                _dict_clust_params["distance_km"] = float(_iter_combos_df.loc[i, 'maxDistance'])

            if _iter_combos_df.loc[i, 'minPts'].astype(int) <= 0:
                print('minPts must be an integer > 0; proceeding with default minPts=3')
                _n_min_cloud_clust_size = 3
            else:
                _dict_clust_params["minimum_samples"] = _iter_combos_df.loc[i, 'minPts'].astype(int)
#                _dict_clust_params["minimum_samples"] = int(_iter_combos_df.loc[i, 'minPts'])

            ''' Validate and assign algorithim based on the clustering name'''
            if _iter_combos_df.loc[i, 'algorithm'] in _dict_algorithms:
                _dict_clust_params["algorithm"] = str(_iter_combos_df.loc[i, 'algorithm'])

            ''' Validate and assign clustering_method based on the clustering name'''
            if _iter_combos_df.loc[i, 'method'] in _dict_clust_method:
                _dict_clust_params["cluster_method"] = str(_iter_combos_df.loc[i, 'method'])

            if _iter_combos_df.loc[i, 'metric'] in _lst_metric:
                _dict_clust_params["metric"] = str(_iter_combos_df.loc[i, 'metric'])

            if isinstance(_iter_combos_df.loc[i, 'weight'],str):
                _dict_clust_params["weight"] = str(_iter_combos_df.loc[i, 'weight'])

            if isinstance(_iter_combos_df.loc[i, 'seed'], str):
                if _iter_combos_df.loc[i, 'seed'] == "random":
                    self._seed = "random"
                    _dict_clust_params["seed"] = np.random
                elif _iter_combos_df.loc[i, 'seed'] == "int":
                    self._seed = "int"
                    _dict_clust_params["seed"] = int
                else:
                    pass

#            if _iter_combos_df.loc[i, 'maxIter'] and _iter_combos_df.loc[i, 'maxIter'] > 0:
            if _iter_combos_df.loc[i, 'maxIter'].astype(int) > 0:
                _dict_clust_params["max_iter"] = int(_iter_combos_df.loc[i, 'maxIter'])

            if _iter_combos_df.loc[i, 'randomState'].astype(int) > 0:
                _dict_clust_params["random_state"] = int(_iter_combos_df.loc[i, 'randomState'])

            if _iter_combos_df.loc[i, 'numClusters'] > 0:
                _dict_clust_params["n_clusters"] = int(_iter_combos_df.loc[i, 'numClusters'])

#            print('Preparing for %s clustering %s with parameters\n%s'
#                  % (_cluster_technique,_s_cloud_clust_name,_dict_clust_params))

        except Exception as err:
            print("Class cluster_quality_metric [get_seq_params] Error message:", err)
            print(traceback.format_exc())

        return _cluster_technique,_s_cloud_clust_name,_dict_clust_params

    ''' Run the cloud or graph clustering sequence for the specific clustering method and parameters '''
    def get_clusters(self,
                     _dict_clust_params,
                      station_df,
                      _cluster_technique: str="cloud",
                      _s_cloud_clust_name: str="DBSCAN"):

        import cloud_clustering as cc
        import graph_clustering as gc
        import numpy as np
        import networkx as nx

        import traceback

#        __st_clust_df = station_df.copy()
        arr_st_coords = station_df[['st_lat','st_lon']].to_numpy()

        try:
            if _cluster_technique == 'cloud':
                cls_clust = cc.cluster_data(_s_cloud_clust_name,**_dict_clust_params)
                labels, labels_true, clust_centers = cls_clust.get_clusters(arr_st_coords)

                if arr_st_coords.shape[0] != labels.shape[0]:
                    raise ValueError('Mismatch in station coordinate and labels array sizes to; cannot proceed')

                station_df['label'] = labels

            elif _cluster_technique == 'graph':
                cls_g_clust = gc.community_detection()
                params = cls_g_clust.set_community_detection_params(_s_cloud_clust_name,**_dict_clust_params)

                ''' G_cluster required to distinguish between communities and valid clusters '''
                G_simple, G_clusters = cls_g_clust.get_communities(station_df)
                station_df['label'] = nx.get_node_attributes(G_simple,'label').values()

            else:
                raise ValueError('Invalid clustering technique: %s' % _cluster_technique)

            ''' Force Regularity of flag is set'''

        except Exception as err:
            print("Class cluster_quality_metric [get_clusters] Error message:", err)
            print(traceback.format_exc())

        return station_df

    ''' Get all quality measures and other parameters for the dataframe with appropriate cluster labels '''
    def get_quality_metrics(self, station_df,lst_graphs):

        import dunn as di
        from sklearn import metrics
        import networkx as nx
        import networkx.algorithms.community as nx_comm
        import numpy as np
        import pandas as pd
        import traceback

        quality_metric_df = pd.DataFrame([])

        try:
#d            _n_num_clust = len(station_df['label'].unique())     # Generated Cluster Count
            _n_num_clust = len([x for x in station_df['label'].unique() if x > -1])     # Generated Cluster Count
            if _n_num_clust <= 1:
                raise ValueError('Cannot compute quality metric for %d clusters' % (_n_num_clust))

            ''' returns the simple graph of the clusters and the set dictionary of cluster nodes '''
            G_simple_, l_G_clusters_ = self.__get_graph_n_labels(station_df)

            _s_st_types = str(station_df['st_type'].unique())   # Station Types
            _n_tot_num_st = station_df.shape[0]     # Station Quantity
            _f_min_dist = self._max_distance        # Minimum Distance
            _n_min_pts = self._minimum_samples      # Minimum Points
            _s_clust = str(self._name)              # Clustering Name
            _s_algo = str(self._algorithm)          # Algorithm
            _s_metric = str(self._metric)           # Metric
            _s_method = str(self._cluster_method)   # Method
            _s_seed = str(self._seed)               # Seed
            __lst_valid_cloud_clust = [frozenset(clust) for clust in l_G_clusters_
                                       if len(clust) >= self._minimum_samples]
            _n_valid_clust = len(__lst_valid_cloud_clust)         # Valid Cluster Count

            # Clustered Station Count
            _n_sts_in_clusters=0
            for x in __lst_valid_cloud_clust:
                _n_sts_in_clusters += len(x)

            _n_noise = station_df.shape[0] - _n_sts_in_clusters   # Unclsutered Noise Count
            _n_avg_deg = sum([d for n, d in G_simple_.degree()
                              if G_simple_.nodes[n]["label"] > -1])/_n_sts_in_clusters # Average Node Degree

            ''' Compute the accuracy of r-regularity constraint on the individual clusters by considering the
                systematic error that is a reproducible inaccuracy consistent for the same clustering strategy.
                For such we apply the weighted mean absolute error to estimate the deviation from the expected degree.
            '''
            sum_deg_abs_err=0
            _deg_wmae=0
            _deg_err_st_count=0
#p            print("\nclusters:",len(lst_graphs))
            for H in lst_graphs:
                H = nx.Graph(H)
                H.remove_nodes_from(list(nx.isolates(H)))
                H.remove_nodes_from([n for n,v in H.nodes(data=True) if v["label"]==-1])
                H_deg_abs_err=0
                _l_deg_diff=[]
                if H.number_of_nodes() > 0:
                    _l_deg_diff = [_n_min_pts-1-d for n, d in H.degree()
                                   if (int(d) < int(_n_min_pts-1) and H.nodes[n]["label"] > -1)]
                if len(_l_deg_diff) > 0:
#p                    print("\ndegree mean absolute error")
#p                    print("minPts:",_n_min_pts)
#p                    print("list deg diff:",_l_deg_diff)
#p                    print("graph nodes:",sorted([d for n,d in H.degree()]))
                    sum_deg_abs_err += sum(_l_deg_diff)
                    _deg_err_st_count += len(_l_deg_diff)
            if _deg_err_st_count > 0:
                _deg_wmae = sum_deg_abs_err/(_deg_err_st_count*(_n_min_pts-1))
#p                print("_deg_wmae", _deg_wmae,_deg_err_st_count)

            ''' prepare valid stations for measuring the quality'''
            lst_st = list(nx.get_node_attributes(G_simple_,'pos').values())
            lst_lbl = list(nx.get_node_attributes(G_simple_,'label').values())

            _f_silhouette = metrics.silhouette_score(lst_st, lst_lbl,
                                                     metric='haversine')   # Silhouette Coefficient
            _f_cal_har = metrics.calinski_harabasz_score(lst_st, lst_lbl)  # Calinski Harabaz score
            _f_dav_bould = metrics.davies_bouldin_score(lst_st, lst_lbl)   # Davies Bouldin score
            _f_dunn = di.dunn_fast(lst_st, lst_lbl)                        # Dunn Index
            _f_modul = nx_comm.modularity(G_simple_,l_G_clusters_)         # Modularity

            try:
                l_conductance = list(nx.conductance(G_simple_, cluster_i, weight='distance')
                                     for cluster_i in __lst_valid_cloud_clust)
                _f_conduct = sum(l_conductance)/len(l_conductance)         # Conductance Average
            except Exception:
                _f_conduct = 0
            _f_cover = nx_comm.coverage(G_simple_, l_G_clusters_)          # Coverage Score
            _f_perform = nx_comm.performance(G_simple_, l_G_clusters_)     # Performance Score

            dict_quality_mesrs = {
                'Station Types': _s_st_types,
                'Station Quantity': _n_tot_num_st,
                'Maximum Distance': _f_min_dist,
                'Minimum Points': _n_min_pts,
                'Name': _s_clust,
                'Algorithm': _s_algo,
                'Metric': _s_metric,
                'Method': _s_method,
                'Seed': _s_seed,
                'Generated Cluster Count': _n_num_clust,
                'Valid Cluster Count': _n_valid_clust,
                'Clustered Station Count': _n_sts_in_clusters,
                'Unclsutered Noise Count': _n_noise,
                'Average Station Degree': _n_avg_deg,
                'Degree Weighted Mean Absolute Error': _deg_wmae,
                'Degree Error Station Count': _deg_err_st_count,
                'Silhouette Coefficient': _f_silhouette,
                'Calinski Harabaz score': _f_cal_har,
                'Davies Bouldin score': _f_dav_bould,
                'Dunn Index': _f_dunn,
                'Modularity': _f_modul,
                'Conductance Average': _f_conduct,
                'Coverage Score': _f_cover,
                'Performance Score': _f_perform,
            }
#            print('Dict qual',dict_quality_mesrs('Seed'))
            quality_metric_df = pd.DataFrame(dict_quality_mesrs, index=[_s_clust])
            quality_metric_df.reset_index(drop=True, inplace=True)

        except Exception as err:
            print("Class cluster_quality_metric [get_quality_metrics] Error message:", err)
#            print(G_simple_.edges('distance'))
            print(traceback.format_exc())

        return quality_metric_df


    def __get_graph_n_labels(self, station_df):

        import sys; sys.path.insert(1, '../lib')
        import graph_clustering as gc
        import networkx as nx
#        import networkx.algorithms.community as nx_comm

        dict_feature_params = {"distance_km":self._max_distance,
                               "minimum_samples":self._minimum_samples}

        cls_g_clust = gc.community_detection(**dict_feature_params)
        G_simple_ = cls_g_clust.get_simple_graph(station_df)
        #print(cloud_G_simple.nodes(data=True))

        _cloud_unique_labels = set(nx.get_node_attributes(G_simple_,'label').values())

        _l_cloud_g_cluster =[]
        for label in _cloud_unique_labels:
            selected_nodes = sorted([n for n,v in G_simple_.nodes(data=True) if v['label'] == label])
            if len(selected_nodes) > 0 and label != -1:
                _l_cloud_g_cluster.append(set(selected_nodes))
            elif len(selected_nodes) > 0 and label == -1:
                for st_node in selected_nodes:
                    _l_cloud_g_cluster.append(set([st_node]))

        return G_simple_, _l_cloud_g_cluster

    ''' get_r_regular_clusters furhter removes those stations and clusters that do not comply with
        minPts and maxDist constraints
    '''
    def get_r_regular_clusters(self,_dict_reg_param,__st_clust_df):

        import sys; sys.path.insert(1, './lib')
        import pandas as pd
        import graph_clustering as gc
        import networkx as nx

        import traceback

        ''' Create subgraphs that comply with r-regularity where r >= minPts-1
            Given that the regularity is based on the average degree, change the scaling value 0.95
            to one that is desired and in the interval (0,1] to set a regularity threshold @_f_reg_thresh
        '''
        try:
            ''' Set the default paramters for the specific r-regularity method '''
            if 'force_regularity' in _dict_reg_param:
                if _dict_reg_param["force_regularity"] in self._l_reg_methods:
                    self._force_regularity=_dict_reg_param["force_regularity"]
                else:
                    raise ValueError('force_regularity must be {%s}'
                                     % str(self._l_reg_methods))

            if 'regularity_threshold' in _dict_reg_param:
                if isinstance(_dict_reg_param["tolerance_scaler"],float) and _dict_reg_param["tolerance_scaler"] < 1.0:
                    self._reg_tol_scaler = _dict_reg_param["tolerance_scaler"]
                else:
                    raise ValueError('regularity_threshold must be %s in invalid and must be in the interval [0,1]'
                                     % str(_dict_reg_param["tolerance_scaler"]))
#d            else:
#d                print('Unspecified regularity_threshold; using default value %0.2f' % (self._reg_tol_scaler))

            ''' (n-1)-simplicies equalant to the regularity required to ensure all target-site stations
                have the minimum required target station connections '''
            _f_reg_thresh = self._reg_tol_scaler*(self._minimum_samples - 1)

            lst_G_simple = []
            ''' Only plot valid clusters '''
            no_noise_df = __st_clust_df[__st_clust_df['label']!= -1]
#            print('%d clusters after removing the noise clusters; i.e. label = -1'
#                  % len(no_noise_df['label'].unique()))

            dict_feature_params = {"distance_km": self._max_distance,
                                   "minimum_samples": self._minimum_samples}
#p            print("Regularity processing maxDist < %0.2f and minPts > %d"
#p                  % (dict_feature_params["distance_km"],
#p                     dict_feature_params["minimum_samples"]-1))
            cls_g_clust = gc.community_detection(**dict_feature_params)
            G_simple = cls_g_clust.get_simple_graph(no_noise_df)
            G_simple.remove_nodes_from(list(nx.isolates(G_simple)))
            if not nx.is_empty(G_simple):
                lst_G_simple = cls_g_clust.get_list_subgraphs(G_simple)
#p            print('\n%d simple subgraphs created after removing clusters with isolated nodes' % len(lst_G_simple))

            ''' remove any graphs with zero average degree '''
            incomplete = True     #flag to start stop while loop
            while incomplete and self._force_regularity != "None":
                incomplete = False
                ''' As a precaution first remove all subgraphs with zero degree nodes; i.e. singletons '''
                for G_idx, G in enumerate(lst_G_simple):
                    if len(G.edges()) == 0:
                        lst_G_simple.pop(G_idx)
#p                        print('...removed subgraph %d with zero degree' % G_idx)
                        incomplete = True

                for G_idx, G in enumerate(lst_G_simple):
#p                    print("\nGraph degree:",G.degree())

                    ''' Average regularity function '''
                    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
                    _avg_degree = sum(degree_sequence)/len(degree_sequence)
                    if self._force_regularity == 'Average' and _avg_degree <= _f_reg_thresh:
                        ''' try to pop if G_idx fails pass and will catch in the next round '''
                        try:
                            lst_G_simple.pop(G_idx)
#p                            print('...removed subgraph %d with average degree %0.02f <= %0.02f tolerated degree'
#p                                  % (G_idx, _avg_degree,_f_reg_thresh))
                            incomplete = True
                        except Exception as err:
                            pass

                    elif self._force_regularity == 'Absolute':
                        ''' Absolute regularity function forces strict minimal regularity '''
                        H = nx.Graph(G)
#                        remove = [n for n,d in dict(H.degree()).items()
#                                  if int(d) < int(self._minimum_samples-1)]
                        remove = [n for n,d in H.degree()
                                  if int(d) < int(self._minimum_samples)]
                        if len(remove) > 0:
#p                            print('...removing nodes %s with degree < %d' % (remove, int(self._minimum_samples-1)))
                            H.remove_nodes_from(remove)
                            if H.number_of_nodes() > 0:
                                lst_G_simple.pop(G_idx)
                                lst_G_simple.append(H)
#p                                print('...replaced subgraph %d with reduced nodes=%d'
#p                                      % (G_idx, H.number_of_nodes()))
                            else:
#p                                print('...removing subgraph %d with %d nodes after node removal'
#p                                      % (G_idx, H.number_of_nodes()))
                                lst_G_simple.pop(G_idx)
                            incomplete = True
                    elif self._force_regularity == 'minPoints':
                        ''' minPoints function remove clusters with size < minimum_samples  '''
                        if G.number_of_nodes() < self._minimum_samples:
                            lst_G_simple.pop(G_idx)

            ''' Modify the station dataframe to reflect the new noise and cluster labels '''
            new_st_clust_df_ = __st_clust_df.copy()

            if self._force_regularity != "None":
                new_st_clust_df_["label"] = -1
                if len(lst_G_simple) > 0:
                    for G_idx, G in enumerate(lst_G_simple):
#                        print(self._minimum_samples)
#                        print("reg fn",[d for n,d in G.degree()])
                        _nodes = sorted([n for n,v in G.nodes(data=True)])
                        new_st_clust_df_.loc[new_st_clust_df_["st_name"].isin(_nodes),"label"] = G_idx

        except Exception as err:
            print("Class cluster_quality_metric [get_r_regular_clusters] Error message:", err)
            print(traceback.format_exc())

        return new_st_clust_df_, lst_G_simple
