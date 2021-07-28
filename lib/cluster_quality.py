#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
    CLASS for community detection in graphs, both spatial and temporal, functions necessary for the station-fault analysis
'''

class cluster_quality_metric():

    def __init__(self):

        self._max_distance=30.0
        self._minimum_samples=3
        self._algorithm = None
        self._metric = None
        self._cluster_method = None
        self._seed = None

        pass

    def set_quality_frame(self, clustering_name: str="greedy_modularity_communities",
                          **metric_params: dict):

        import traceback
        import numpy as np

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

            #if 'seed' in metric_params:
            #    if isinstance(metric_params["seed"], np.random):
            #        self._seed="random"
            #        print("\n",self._seed,"\n")
            #    elif isinstance(metric_params["seed"], int):
            #        print("\n",self._seed,"\n")
            #    else:
            #        raise ValueError('Seed %s is invalid.' % (metric_params["seed"]))

        except Exception as err:
            print("Class cluster_quality_metric [set_quality_frame] Error message:", err)
            print(traceback.format_exc())

        return self


    def get_seq_params(self, _iter_combos_df, exp_seq: int = 0):

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

        '''
        _l_cluster_techniques = ['cloud','graph']
        _l_cloud_cluster_name = ['DBSCAN','HDBSCAN','AFFINITYPROPAGATION','OPTICS','MEANSHIFT',
                                  'AGGLOMERATIVE','BIRCH','KMEANS','KNN','DENCLUE']
        _l_graph_cluster_name = ['GREEDY','NAIVE-GREEDY','LPC','ASYNC-LPA',
                                 'LUKES','ASYNC-FLUID','GIRVAN-NEWMAN']
        _dict_algorithms = {"auto", "ball_tree", "kd_tree", "brute",
                            "best","generic","prims_kdtree","prims_balltree","boruvka_kdtree","boruvka_balltree"}

        _dict_clust_method = {"leaf","eom","xi","dbscan"}
        '''In all instances when possible <haversine> will be the choice; else it will be <precomputed>'''
        _lst_metric = ['haversine','euclidean','manhattan','minkowski']

        _dict_clust_params = {}

        i=exp_seq
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
                _dict_clust_params["minimum_samples"] = int(_iter_combos_df.loc[i, 'minPts'])

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

        _cloud_clust_st_df = station_df.copy()
        arr_st_coords = _cloud_clust_st_df[['st_lat','st_lon']].to_numpy()

        try:
            if _cluster_technique == 'cloud':
                cls_clust = cc.cluster_data(_s_cloud_clust_name,**_dict_clust_params)
                labels, labels_true, clust_centers = cls_clust.get_clusters(arr_st_coords)

                if arr_st_coords.shape[0] != labels.shape[0]:
                    raise ValueError('Mismatch in station coordinate and labels array sizes to; cannot proceed')

                _cloud_clust_st_df['label'] = labels

            elif _cluster_technique == 'graph':
                cls_g_clust = gc.community_detection()
                params = cls_g_clust.set_community_detection_params(_s_cloud_clust_name,**_dict_clust_params)

                ''' G_cluster required to distinguish between communities and valid clusters '''
                G_simple, G_clusters = cls_g_clust.get_communities(_cloud_clust_st_df)
                _cloud_clust_st_df['label'] = nx.get_node_attributes(G_simple,'label').values()

            else:
                raise ValueError('Invalid clustering technique: %s' % _cluster_technique)

        except Exception as err:
            print("Class cluster_quality_metric [get_clusters] Error message:", err)
            print(traceback.format_exc())

        return _cloud_clust_st_df

    ''' Get all quality measures and other parameters for the dataframe with appropriate cluster labels '''
    def get_quality_metrics(self, station_df):

        import dunn as di
        from sklearn import metrics
        import networkx as nx
        import networkx.algorithms.community as nx_comm
        import numpy as np
        import pandas as pd
        import traceback

        quality_metric_df = pd.DataFrame([])

        try:
            _n_num_clust = len(station_df['label'].unique())     # Generated Cluster Count
            if _n_num_clust <= 1:
                raise ValueError('Cannot compute quality metric for %d clusters' % (_n_num_clust))

            cloud_G_simple_, l_cloud_g_cluster_ = self.__get_graph_n_labels(station_df)
#            print(cloud_G_simple_.nodes(data=True))

            _s_st_types = str(station_df['st_type'].unique())   # Station Types
            _n_tot_num_st = station_df.shape[0]     # Station Quantity
            _f_min_dist = self._max_distance        # Minimum Distance
            _n_min_pts = self._minimum_samples      # Minimum Points
            _s_clust = str(self._name)              # Clustering Name
            _s_algo = str(self._algorithm)          # Algorithm
            _s_metric = str(self._metric)           # Metric
            _s_method = str(self._cluster_method)   # Method
            _s_seed = str(self._seed)               # Seed
#            print('seed is',_s_seed)
            __lst_valid_cloud_clust = [frozenset(clust) for clust in l_cloud_g_cluster_
                                       if len(clust) >= self._minimum_samples]
            _n_valid_clust = len(__lst_valid_cloud_clust)         # Valid Cluster Count
            # Clustered Station Count
            _n_sts_in_clusters = len([v['label'] for n,v in cloud_G_simple_.nodes(data=True) if v['label'] > -1])
            _n_noise = station_df.shape[0] - _n_sts_in_clusters   # Unclsutered Noise Count
            _n_avg_deg = sum([v for k, v in cloud_G_simple_.degree()
                              if cloud_G_simple_.nodes[k]["label"] > -1])/_n_sts_in_clusters # Average Node Degree
            _f_silhouette = metrics.silhouette_score(station_df[['st_lat','st_lon']].to_numpy(),
                                                     list(station_df['label']),
                                                     metric='haversine')               # Silhouette Coefficient
            _f_cal_har = metrics.calinski_harabasz_score(station_df[['st_lat','st_lon']].to_numpy(),
                                                         list(station_df['label']))    # Calinski Harabaz score
            _f_dav_bould = metrics.davies_bouldin_score(station_df[['st_lat','st_lon']].to_numpy(),
                                                        list(station_df['label']))     # Davies Bouldin score
            _f_dunn = di.dunn_fast(station_df[['st_lat','st_lon']].to_numpy(),
                                   list(station_df['label']))                           # Dunn Index
            _f_modul = nx_comm.modularity(cloud_G_simple_,l_cloud_g_cluster_)           # Modularity

            try:
                l_conductance = list(nx.conductance(cloud_G_simple_, cluster_i, weight='distance')
                                     for cluster_i in __lst_valid_cloud_clust)
                _f_conduct = sum(l_conductance)/len(l_conductance)                      # Conductance Average
            except Exception:
                _f_conduct = 0
            _f_cover = nx_comm.coverage(cloud_G_simple_, l_cloud_g_cluster_)            # Coverage Score
            _f_perform = nx_comm.performance(cloud_G_simple_, l_cloud_g_cluster_)       # Performance Score

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
                'Average Node Degree': _n_avg_deg,
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
#            print(cloud_G_simple_.edges('distance'))
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
        cloud_G_simple_ = cls_g_clust.get_simple_graph(station_df)
        #print(cloud_G_simple.nodes(data=True))

        _cloud_unique_labels = set(nx.get_node_attributes(cloud_G_simple_,'label').values())

        _l_cloud_g_cluster =[]
        for label in _cloud_unique_labels:
            selected_nodes = sorted([n for n,v in cloud_G_simple_.nodes(data=True) if v['label'] == label])
            if len(selected_nodes) > 0 and label != -1:
                _l_cloud_g_cluster.append(set(selected_nodes))
            elif len(selected_nodes) > 0 and label == -1:
                for st_node in selected_nodes:
                    _l_cloud_g_cluster.append(set([st_node]))

        return cloud_G_simple_, _l_cloud_g_cluster