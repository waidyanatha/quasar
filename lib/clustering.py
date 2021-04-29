#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
    CLASS for clustering of data, both spatial and temporal, functions necessary for the station-fault analysis
'''

class cluster_data():

    def __init__(self, clustering_name="DBSCAN", **cluster_params):

        _default_distance = 50.0
        _cluster_method_name = ['DBSCAN','HDBSCAN','OPTICS','MEANSHIFT','KMEANS','KNN', 'DENCLUE']
        _lst_metric = ['haversine','euclidean','manhattan','minkowski']
        _lst_algo = ['auto', 'ball_tree', 'kd_tree', 'brute']
        _lst_clust_method = ['xi','dbscan']

        self.name = clustering_name
        self.epsilon=_default_distance/6371.0088
        self.minimum_samples = 3
        self.cluster_std=0.4
        self.n_clusters=5
        self.random_state=0
        self.maximum_iterations=300
        self.centroid_init=5
        self.algorithm='ball_tree'
        self.metric='haversine'
        self.cluster_method='xi'
        self.fit_predict = True
        self.gen_min_span_tree=True
        self.prediction_data=True

        try:
            ''' Set the default paramters for the specific clustering method '''
            if self.name not in _cluster_method_name:
                raise ValueError('{0} is an undefined clustering_name. Must be {1}'.format(self.name,_cluster_method_name))

            if 'distance_km' in cluster_params:
                if isinstance(cluster_params["distance_km"],float) and cluster_params["distance_km"] > 0:
                    self.epsilon=cluster_params["distance_km"]/6371.0088
                else:
                    raise ValueError('distance_km %s must be a float > 0.'
                                     % str(cluster_params["distance_km"]))

            if 'minimum_samples' in cluster_params:
                if isinstance(cluster_params["minimum_samples"],int) and cluster_params["minimum_samples"] > 0:
                    self.minimum_samples=cluster_params["minimum_samples"]
                else:
                    raise ValueError('minimum_samples %s must be an int > 0.'
                                     % str(cluster_params["minimum_samples"]))

            if 'max_iter' in cluster_params:
                if isinstance(cluster_params["max_iter"],int) and cluster_params["max_iter"] > 0:
                    self.maximum_iterations=cluster_params["max_iter"]
                else:
                    raise ValueError('minimum_samples %s must be an int > 0.'
                                     % str(cluster_params["max_iter"]))

            if 'random_state' in cluster_params:
                if isinstance(cluster_params["random_state"],int) and cluster_params["random_state"] > 0:
                    self.random_state=cluster_params["random_state"]
                else:
                    print('random_state')
                    raise ValueError('minimum_samples %s must be an int > 0.'
                                     % str(cluster_params["random_state"]))

            if 'algorithm' in cluster_params:
                if cluster_params["algorithm"] in _lst_algo:
                    self.algorithm=cluster_params["algorithm"]
                else:
                    raise ValueError('algorithm {0} is invalid must be {1}. Continue with default value [{2}]'.
                                     format(cluster_params["algorithm"],_lst_algo,self.algorithm))

            if 'metric' in cluster_params:
                if cluster_params["metric"] in _lst_metric:
                    self.metric=cluster_params["metric"]
                else:
                    raise ValueError('metric {0} is invalid must be {1}. Continue with default value [{2}]'.
                                     format(cluster_params["metric"],_lst_metric,self.metric))

            if 'cluster_method' in cluster_params:
                if cluster_params["cluster_method"] in _lst_clust_method:
                    self.cluster_method=cluster_params["cluster_method"]
                else:
                    raise ValueError('cluster_method {0} is invalid must be {1}. Continue with default value [{2}]'.
                                     format(cluster_params["cluster_method"],_lst_clust_method,self.cluster_method))
            if 'n_clusters' in cluster_params:
                if isinstance(cluster_params["n_clusters"],int) and cluster_params["n_clusters"] > 0:
                    self.n_clusters=cluster_params["n_clusters"]
                else:
                    raise ValueError('n_clusters %s must be an int > 0.'
                                     % str(cluster_params["n_clusters"]))

            if 'fit_predict' in cluster_params:
                if isinstance(cluster_params["fit_predict"],bool):
                    self.fit_predict=cluster_params["fit_predict"]
                else:
                    raise ValueError('fit_predict %s is invalid must be a bool TRUE/FALSE.'
                                     % str(cluster_params["fit_predict"]))

        except Exception as err:
            print("[cluster_data] Error message:", err)

    '''
        Get cluster labels for a given clustering method
    '''
    def get_clusters(self,st_arr):

        import numpy as np
        from sklearn.cluster import DBSCAN, KMeans, OPTICS, MeanShift
        import hdbscan
#        import sklearn.utils
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_blobs
        import sys
        sys.path.insert(1, '../lib')
        import denclue


        if self.name == 'DBSCAN':
            clusterer = DBSCAN(eps=self.epsilon,
                               min_samples=self.minimum_samples,
                               algorithm=self.algorithm,
                               metric=self.metric)
        elif self.name == 'HDBSCAN':
            clusterer = hdbscan.HDBSCAN(min_cluster_size=self.minimum_samples,
                                        cluster_selection_epsilon = self.epsilon,
                                        metric=self.metric,
                                        gen_min_span_tree=True,
                                        prediction_data=True)
        elif self.name == 'OPTICS':
            clusterer = OPTICS(min_samples=self.minimum_samples,
                               metric=self.metric,
                               cluster_method=self.cluster_method,
                               algorithm=self.algorithm)
        elif self.name == 'MEANSHIFT':
            clusterer = MeanShift()

        elif self.name == 'KMEANS':
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(st_arr)
            ''' init="random" or "k-means++"
                n_init=10 (Number of runs with different centroid seeds)
                max_iter=300 (Maximum number of iterations for a single run)
                random_state=5 (Determines random number generation for centroid initialization)
            '''
            clusterer = KMeans(init='k-means++',
                               n_clusters=self.n_clusters,
                               n_init=self.centroid_init,
                               max_iter=self.maximum_iterations,
                               random_state=self.random_state)
        elif self.name == 'KNN':
            print('TBD')

        elif self.name == 'DENCLUE':
            clusterer = denclue.DENCLUE(h=None,
                                        eps=self.epsilon,
                                        min_density=0.,
                                        metric=self.metric)
            if self.fit_predict:
                print('WARNING DENCLUE does does not have a fit_predict function. Switching to fit')
                self.fit_predict = False

        else:
            print("something was not right")

        X, _labels_true = make_blobs(n_samples=len(st_arr),
                                    centers=st_arr,
                                    cluster_std=self.cluster_std,
                                    random_state=self.random_state)

        if self.fit_predict:
            clusterer.fit_predict(np.radians(st_arr))
        else:
            clusterer.fit(np.radians(st_arr))

#        _core_samples_mask = np.zeros_like(clusterer.labels_, dtype=bool)
#        _core_samples_mask[clusterer.core_sample_indices_] = True

        print(clusterer)

        cluster_centers = self.get_cluster_centers(self.name,clusterer)

        return clusterer.labels_, _labels_true, cluster_centers #, _core_samples_mask

    '''
        Get Cluster Centers
    '''
    def get_cluster_centers(self,name,clusters):

        if name == 'DBSCAN':
            return None
        elif name == 'MEANSHIFT':
            return clusters.cluster_centers_
        elif name == 'DENCLUE':
            return clusters.clust_info_[0]['centroid']
        else:
            return None

    '''
        K nearest neigbour clustering
        Used in STATION FAULT clustering
    '''
    def get_nn_labels(self,st_flt_list):

        from sklearn.neighbors import NearestNeighbors

        # Augment station array with cluster number
        # Start a new station coorinates and details tuple
        st_list = []
        i=0
        for i in range(len(labels)):
            st_row = [tmp_arr[i,0],labels[i],tmp_arr[i,1],tmp_arr[i,2],tmp_arr[i,3]]
            st_list.append(list(st_row))

        clusters = list({item[1] for item in st_list})

        for each_cluster in clusters:
            cluster_list = list(st_list[j] for j in range(len(st_list)) if st_list[j][1] == each_cluster)
            cluster_arr = np.delete(cluster_list, [0,1,4],axis=1).astype(np.float)
            nbrs = NearestNeighbors(n_neighbors=3, algorithm='brute', metric='haversine').fit(cluster_arr)
            distances, indices = nbrs.kneighbors(cluster_arr)
            print(nbrs.kneighbors_graph(cluster_arr).toarray())

            each_cluster_clique = client.get_stations(latitude=-42.693,longitude=173.022,maxradius=30.0/6371.0, starttime = "2016-11-13 11:05:00.000",endtime = "2016-11-14 11:00:00.000")
            print(each_cluster_clique)
            _=inventory.plot(projection="local")

            break

        sorted_rank = sorted(st_list, key=lambda i: (int(i[1])), reverse=True)
        #print('Code, Cluster, Latitude, Longitude, Elevation')
        #print(sorted_rank)
        return sorted_rank

    '''
        K Means clustering - station-fault distance metric
        Parameters:
            number of clusters = 5 gives optimal Homogeneity, V-measure, and Silhouette Coefficient
            maximum number of iterations = 300 to minimize clustering quality; i.e. sum of the squared error
    '''
    def get_kmean_labels(self, st_flt_arr, n_clusters=5):

        from sklearn.cluster import KMeans
#        import sklearn.utils
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_blobs
        import numpy as np

        ''' make station-faults blob with shape of X being 6 features and len(st-flt-arr) '''
#        X, labels_true = make_blobs(n_samples=len(st_flt_arr), centers=st_flt_arr, cluster_std=0.4,random_state=0)
        scaler = StandardScaler()
#        scaled_features = scaler.fit_transform(X)
        scaled_features = scaler.fit_transform(st_flt_arr)
        ''' init = "random", "k-means++" "'''
        kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=5,max_iter=300, random_state=5)
        ''' use either fit_predict or fit - using fit because it works with scaled features '''
        #label = kmeans.fit_predict(scaled_features)
        kmeans.fit(scaled_features)
        y_kmeans = kmeans.predict(scaled_features)
        s_stdout = "Statistics from the initialization run with the lowest SSE;\n"
        s_stdout += "Inertia {0} with {1} iterations befor staturation and {3} \ncenters\n {2}"
        print(s_stdout.format(kmeans.inertia_, kmeans.n_iter_, kmeans.cluster_centers_, len(kmeans.cluster_centers_)))
        labels = kmeans.labels_
        print("\nThe station and fault K-means clustering {1} labels \n{0}".format(kmeans.labels_, len(kmeans.labels_)))
#        core_samples_mask = np.zeros_like(kmeans.labels_, dtype=bool)
#        core_samples_mask[kmeans.core_sample_indices_] = True

#        return kmeans
#        return labels, labels_true, core_samples_mask
#        return labels, labels_true, kmeans.cluster_centers_, scaled_features, y_kmeans
        return kmeans.labels_, kmeans.cluster_centers_, scaled_features, y_kmeans