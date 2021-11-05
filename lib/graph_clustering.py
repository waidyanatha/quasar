#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
    CLASS for community detection in graphs, both spatial and temporal, functions necessary for the station-fault analysis
'''

class community_detection():

#    def __init__(self, clustering_name="greedy_modularity_communities", **cluster_params):
    def __init__(self, **feature_params):

        _default_distance = 30.0
        self.max_distance=_default_distance
#        self.epsilon=_default_distance/6371.0088
        self.minimum_samples=3

        try:
            if 'distance_km' in feature_params:
                if isinstance(feature_params["distance_km"],float) and feature_params["distance_km"] > 0:
                    self.epsilon=feature_params["distance_km"]/6371.0088
                    self.max_distance=feature_params["distance_km"]
                else:
                    raise ValueError('distance_km %s must be a float > 0.'
                                     % str(feature_params["distance_km"]))

            if 'minimum_samples' in feature_params:
                if isinstance(feature_params["minimum_samples"],int) and feature_params["minimum_samples"] > 0:
                    self.minimum_samples=feature_params["minimum_samples"]
                else:
                    raise ValueError('minimum_samples %s must be an int > 0.'
                                     % str(feature_params["minimum_samples"]))

        except Exception as err:
            print("Class community_detection [__init__()] Error message:", err)


    def set_community_detection_params(self, clustering_name="greedy_modularity_communities", **cluster_params):

        import traceback
        import numpy as np

        _default_distance = 50.0
        _cluster_method_name = ['ASYNC-LPA',  #asyn_lpa_communities
                                'LPC',        #label_propagation_communities'
                                'GREEDY',     #greedy_modularity_communities
                                'NAIVE-GREEDY',  #_naive_greedy_modularity_communities'
                                'LUKES',         #lukes_partitioning
                                'ASYNC-FLUID',   #asyn_fluidc
                                'GIRVAN-NEWMAN'  #girvan_newman
                                ]

        self.name=clustering_name
#        self.max_distance=_default_distance
#        self.epsilon=_default_distance/6371.0088
#        self.minimum_samples=3
        self.seed=None
        self.weight='distance'     # necessary for defining the attribute that holds the edgeweight
        self.maximum_node_weight=200
        self.maximum_iterations=200

        try:
            ''' Set the default paramters for the specific clustering method '''
            if self.name not in _cluster_method_name:
                raise ValueError('{0} is an undefined graph clustering_name. Must be {1}'
                                 .format(self.name,_cluster_method_name))

            if 'distance_km' in cluster_params:
                if isinstance(cluster_params["distance_km"],float) and cluster_params["distance_km"] > 0:
#                if cluster_params["distance_km"].astype(float) > 0:
                    self.epsilon=cluster_params["distance_km"]/6371.0088
                    self.max_distance=cluster_params["distance_km"]
                else:
                    raise ValueError('distance_km %s must be a float > 0.'
                                     % str(cluster_params["distance_km"]))

            if 'minimum_samples' in cluster_params:
                if cluster_params["minimum_samples"] > 0:
#                if isinstance(cluster_params["minimum_samples"],int) and cluster_params["minimum_samples"] > 0:
#                if cluster_params["minimum_samples"].astype(int) > 0:
                    self.minimum_samples=cluster_params["minimum_samples"]
                else:
                    raise ValueError('minimum_samples %s must be an int > 0.'
                                     % str(cluster_params["minimum_samples"]))

            if 'seed' in cluster_params:
#                print('checking seed',cluster_params["seed"])
                self.seed=cluster_params["seed"]
#                if (
#                    isinstance(cluster_params["seed"],(int,np.random,None))
#                    isinstance(cluster_params["seed"], np.random) or     #fix this check it's not working
#                    cluster_params["seed"] == None):
#                ):
#                if cluster_params["seed"] in ['int', np.random, None]:
#                    self.seed=cluster_params["seed"]
#                else:
#                    raise ValueError('seed parameter %s must be {int, np.random, None}.'
#                                     % str(cluster_params["seed"]))

            if 'weight' in cluster_params:
                if (cluster_params["weight"] == 'distance'):
                    self.weight=cluster_params["weight"]
                else:
                    raise ValueError('weight parameter %s must be = {distance}.'
                                     % str(cluster_params["weight"]))

            if 'max_iter' in cluster_params:
                if isinstance(cluster_params["max_iter"],int) and cluster_params["max_iter"] > 0:
                    self.maximum_iterations=cluster_params["max_iter"]
                else:
                    raise ValueError('maximum iteration %s must be an int > 0.'
                                     % str(cluster_params["max_iter"]))

        except Exception as err:
            print("Class community_detection [set_community_detection_params] Error message:", err)
            print(traceback.format_exc())

        return self

    def get_communities(self,station_df):

        import networkx as nx
        import networkx.algorithms.community as nx_comm

        try:
            ''' sample the graph '''
            g_simple_ = self.get_simple_graph(station_df)

            if nx.is_empty(g_simple_):
                raise ValueError('A simple graph with %d stations was not created' % station_df.shape[0])

            if self.name == 'ASYNC-LPA':     #asyn_lpa_communities
                g_communities_ = list(nx_comm.asyn_lpa_communities(
                    g_simple_,
                    weight=self.weight,
                    seed=self.seed))

            elif self.name == 'LPC':     #label_propagation_communities
                g_communities_ = list(nx_comm.label_propagation_communities(
                    g_simple_))

            elif self.name == 'GREEDY':     # greedy_modularity_communities
                g_communities_ = list(nx_comm.greedy_modularity_communities(
                    g_simple_))

            elif self.name == 'NAIVE-GREEDY':     #_naive_greedy_modularity_communities
                g_communities_ = list(nx_comm._naive_greedy_modularity_communities(
                    g_simple_))

            elif self.name == 'LUKES':     # lukes_partitioning
                # TODO: create MST of g_simple first but removing the mimum weigted edge doesn't seem right
                g_communities_ = list(nx_comm.lukes_partitioning(
                    g_simple_,
                    edge_weight=self.weight,
                    max_size=self.maximum_node_weight))

            elif self.name == 'ASYNC-FLUID':     # asyn_fluidc
                # TODO: create complete graph for g_simple but a complete graph would not work
                g_communities_ = list(nx_comm.asyn_fluidc(
                    g_simple_,
                    k=15,
                    max_iter=300,
                    seed=self.seed))

            elif self.name == 'GIRVAN-NEWMAN':     # girvan_newman
#                g_communities_ = list(nx_comm.girvan_newman(g_simple_))
#                tmp_communities = nx_comm.girvan_newman(g_simple_)
#                g_communities_  = next(tmp_communities)
                g_communities_  = list(next(nx_comm.girvan_newman(g_simple_)))
#                print(list(g_communities_))

            else:
                raise AttributeError("something was not right")

#            g_simple_ = self.set_graph_cluster_labels(g_simple_, g_communities_)
            if isinstance(g_communities_, list): #len(g_communities_)>0
                g_simple_ = self.set_graph_cluster_labels(g_simple_, g_communities_)

            return g_simple_, g_communities_

        except Exception as err:
            print("Class community_detection [get_communities] Error message:", err)


    def set_graph_cluster_labels(self, _g_simple, _g_communities):

        import networkx as nx
#        import matplotlib.pyplot as plt

        ''' remove all clusters less than minimum_sample '''
#        print('Relabelling communities and removing those < %d' % self.minimum_samples)
        label_val = 0
        for cl_idx, cl_nodes_dict in enumerate(_g_communities):
#            if len(cl_nodes_dict) >= self.minimum_samples:
            if len(cl_nodes_dict) > 1:
                node_attr_dict = dict.fromkeys(cl_nodes_dict, label_val)
                nx.set_node_attributes(_g_simple, node_attr_dict, "label")
                label_val +=1
        else:
            node_attr_dict = dict.fromkeys(cl_nodes_dict, -1)
            nx.set_node_attributes(_g_simple, node_attr_dict, "label")
        '''
        for cl_idx, cl_nodes_dict in enumerate(_g_communities):
            node_attr_dict = dict.fromkeys(cl_nodes_dict, cl_idx)
            nx.set_node_attributes(_g_simple, node_attr_dict, "label")
        '''
        return _g_simple

#    def get_simple_graph(self,station_coordinates):
    def get_simple_graph(self,station_df):

        import numpy as np
        import networkx as nx
        from sklearn.metrics.pairwise import haversine_distances

        station_df.reset_index(drop=True, inplace=True)
        g_simple = nx.Graph(name='All-Stations-Simple-Graph') # Simple graph

        lat = np.array(station_df['st_lat'])
        lon = np.array(station_df['st_lon'])
        st_coords = np.column_stack((lat,lon))
        dist_arr = haversine_distances(np.radians(st_coords),np.radians(st_coords))

        for i in range(dist_arr.shape[0]):
#            g_simple.add_node(i,pos=(clust_st[i,1],clust_st[i,0]),label=-1)
            g_simple.add_node(station_df.loc[i,'st_name'],
                              pos=(station_df.loc[i,'st_lat'],
                                   station_df.loc[i,'st_lon']),
                              label=station_df.loc[i,'label'],
                              station=station_df.loc[i,'st_name'])
            for j in range(i,dist_arr.shape[1]):
                if j>i:
                    __f_dist = round(dist_arr[i,j]*6371.088,2)
                    ''' Build the simple graph with edge weights <= 30 Km '''
                    if __f_dist <= self.max_distance:
#                        g_simple.add_edge(i,j,distance=__f_dist)
                        g_simple.add_edge(station_df.loc[i,'st_name'],
                                          station_df.loc[j,'st_name'],
                                          distance=round(__f_dist,2))
        ''' remove nodes with no neigbours <= _l_max_distance '''
#        g_simple.remove_nodes_from(list(nx.isolates(g_simple)))

        return g_simple

    ''' Create a list of community subgraphs '''
    def get_list_subgraphs(self,simple_graph):

        import networkx as nx

        l_sub_graphs = []
        #unique_labels = list(set([label for n,label in simple_graph.nodes.data("label")]))
        unique_labels = set(nx.get_node_attributes(simple_graph,'label').values())
        ''' remove noise label = -1 from set
            TODO: only execute of -1 labels exist
        '''
        if -1 in unique_labels:
            unique_labels.remove(-1)

        for label in unique_labels:
            selected_nodes = sorted([n for n,v in simple_graph.nodes(data=True) if v['label'] == label])
            if len(selected_nodes) > 0:
                l_sub_graphs.append(simple_graph.subgraph(selected_nodes))
        return l_sub_graphs