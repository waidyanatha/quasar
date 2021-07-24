#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
    CLASS for community detection in graphs, both spatial and temporal, functions necessary for the station-fault analysis
'''

class cluster_quality_metric():

    def __init__(self):
        pass

    def set_quality_frame(self, clustering_name: str="greedy_modularity_communities",
                          **metric_params: dict):

        import traceback
        import numpy as np

        self._name=clustering_name
        self._max_distance=30.0
        self._minimum_samples=3
        self._algorithm = None
        self._metric = None
        self._cluster_method = None

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

        except Exception as err:
            print("Class cluster_quality_metric [set_quality_frame] Error message:", err)
            print(traceback.format_exc())

        return self

    ''' Get all quality measures and other parameters for the dataframe with appropriate cluster labels '''
    def get_quality_metrics(self, station_df):

        import dunn as di
        from sklearn import metrics
        import networkx as nx
        import networkx.algorithms.community as nx_comm
        import numpy as np
        import pandas as pd

        cloud_G_simple_, l_cloud_g_cluster_ = self.__get_graph_n_labels(station_df)

        _s_st_types = str(station_df['st_type'].unique())   # Station Types
        _n_tot_num_st = station_df.shape[0]     # Station Quantity
        _f_min_dist = self._max_distance        # Minimum Distance
        _n_min_pts = self._minimum_samples      # Minimum Points
        _s_clust = str(self._name)              # Clustering Name
        _s_algo = str(self._algorithm)          # Algorithm
        _s_metric = str(self._metric)           # Metric
        _s_method = str(self._cluster_method)    # Method
        _n_num_clust = len(station_df['label'].unique())     # Generated Cluster Count
        __lst_valid_cloud_clust = [frozenset(clust) for clust in l_cloud_g_cluster_
                           if len(clust) >= self._minimum_samples]
        _n_valid_clust = len(__lst_valid_cloud_clust)   # Valid Cluster Count
        _n_sts_in_clusters = len([v['label'] for n,v in cloud_G_simple_.nodes(data=True) if v['label'] > -1])   # Clustered Station Count
        _n_noise = station_df.shape[0] - _n_sts_in_clusters   # Unclsutered Noise Count
        _n_avg_deg = sum([v for k, v in cloud_G_simple_.degree()
                  if cloud_G_simple_.nodes[k]["label"] > -1])/_n_sts_in_clusters # Average Node Degree
        _f_silhouette = metrics.silhouette_score(station_df[['st_lat','st_lon']].to_numpy(),
                                                 list(station_df['label']),
                                                 metric='haversine')                      # Silhouette Coefficient
        _f_cal_har = metrics.calinski_harabasz_score(station_df[['st_lat','st_lon']].to_numpy(),
                                             list(station_df['label']))   # Calinski Harabaz score
        _f_dav_bould = metrics.davies_bouldin_score(station_df[['st_lat','st_lon']].to_numpy(),
                                            list(station_df['label']))    # Davies Bouldin score
        _f_dunn = di.dunn_fast(station_df[['st_lat','st_lon']].to_numpy(),
                       list(station_df['label']))          # Dunn Index
        _f_modul = nx_comm.modularity(cloud_G_simple_,l_cloud_g_cluster_)   # Modularity
        l_conductance = list(nx.conductance(cloud_G_simple_, cluster_i, weight='distance')
                     for cluster_i in __lst_valid_cloud_clust)
        _f_conduct = sum(l_conductance)/len(l_conductance)                     # Conductance Average
        _f_cover = nx_comm.coverage(cloud_G_simple_, l_cloud_g_cluster_)        # Coverage Score
        _f_perform = nx_comm.performance(cloud_G_simple_, l_cloud_g_cluster_)   # Performance Score

        dict_quality_mesrs = {
            'Station Types': _s_st_types,
            'Station Quantity': _n_tot_num_st,
            'Minimum Distance': _f_min_dist,
            'Minimum Points': _n_min_pts,
            'Name': _s_clust,
            'Algorithm': _s_algo,
            'Metric': _s_metric,
            'Method': _s_method,
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
        quality_metric_df = pd.DataFrame(dict_quality_mesrs, index=[_s_clust])
        quality_metric_df.reset_index(drop=True, inplace=True)

        return quality_metric_df

    def deprecated_get_quality_metrics(self, station_df):

        import dunn as di
        from sklearn import metrics
        import networkx as nx
        import networkx.algorithms.community as nx_comm
        import numpy as np
        import pandas as pd

        cloud_G_simple_, l_cloud_g_cluster_ = self.__get_graph_n_labels(station_df)

        _df_cols = ['Station Types',          #0
                    'Station Quantity',      #1
                    'Minimum Distance',      #2
                    'Minimum Points',        #3
                    'Name', 'Algorithm',     #4
                    'Metric',                #5
                    'Method',                #6
                    'Generated Cluster Count',  #7
                    'Valid Cluster Count',      #8
                    'Clustered Station Count',  #9
                    'Unclsutered Noise Count',  #10
                    'Average Node Degree',      #11
                    'Silhouette Coefficient',   #12
                    'Calinski Harabaz score',   #13
                    'Davies Bouldin score',     #14
                    'Dunn Index',               #15
                    'Modularity',               #16
                    'Conductance Average',      #17
                    'Coverage Score',           #18
                    'Performance Score'         #19
                   ]
        quality_metric_df = pd.DataFrame(columns = _df_cols)

        ''' Quality metric metadata '''
        _s_st_types = station_df['st_type'].unique()
        _n_tot_num_st = station_df.shape[0]

        ''' Generate clustering statistics '''
        _n_community_nodes = sum([len(v) for v in cloud_G_simple_])
        _n_min_size_clusters = len(set([v['label'] for n,v in cloud_G_simple_.nodes(data=True) if v['label'] > -1]))
        _n_nodes_in_clusters = len([v['label'] for n,v in cloud_G_simple_.nodes(data=True) if v['label'] > -1])
        _n_noise_count = station_df.shape[0] - _n_nodes_in_clusters
        _n_avg_degree = sum([v for k, v in cloud_G_simple_.degree()
                             if cloud_G_simple_.nodes[k]["label"] > -1])/_n_nodes_in_clusters

        ''' total number of stations submitted for clustering '''

        quality_metric_df['Name'] = self._name
        ''' Minimum size of a cluster; i.e. minPts '''
#        quality_metric_df['Station Quantity'] = station_df.shape[0]
        ''' clustering technique name '''
#        quality_metric_df['Minimum Points'] = self._minimum_samples
        ''' Minimum cluster density distance in Km '''
#        quality_metric_df['Minimum Distance'] = self._max_distance
        '''Number of clusters wth size <= distance '''
#        quality_metric_df['Generated Cluster Count'] = len(l_cloud_g_cluster_)
        '''Number of method generated communities: '''
#        quality_metric_df['Valid Cluster Count'] = _n_min_size_clusters
        '''Number of total nodes in valid clusters'''
#        quality_metric_df['Clustered Station Count'] = _n_nodes_in_clusters
        ''' Number of noise points (total stations - valid cluster nodes) '''
#        quality_metric_df['Unclsutered Noise Count'] = _n_noise_count
        '''Estimated average degree of valid cluster nodes '''
#        quality_metric_df['Average Node Degree'] = _n_avg_degree
        ''' Conductance does not accept singelton clusters '''
        __lst_valid_cloud_clust = [frozenset(clust) for clust in l_cloud_g_cluster_
                                   if len(clust) >= self._minimum_samples]
        l_conductance = list(nx.conductance(cloud_G_simple_, cluster_i, weight='distance')
                             for cluster_i in __lst_valid_cloud_clust)
        ''' Conductance average [-1.0,1.0]: %0.4f '''
#        quality_metric_df['Conductance Average'] = sum(l_conductance)/len(l_conductance)
        ''' Modularity score [-1.0,1.0]: %0.4f '''
#        quality_metric_df['Modularity'] = nx_comm.modularity(cloud_G_simple_,l_cloud_g_cluster_)
        ''' Coverage score [-1.0,1.0]: %0.4f '''
#        quality_metric_df['Coverage Score'] = nx_comm.coverage(cloud_G_simple_, l_cloud_g_cluster_)
        ''' Performance score [-1.0,1.0]: %0.4f '''
#        quality_metric_df['Performance Score'] = nx_comm.performance(cloud_G_simple_, l_cloud_g_cluster_)

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