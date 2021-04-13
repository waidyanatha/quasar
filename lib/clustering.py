#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
    CLASS for clustering of data, both spatial and temporal, functions necessary for the station-fault analysis
'''
class cluster_data():

    def __init__(self, method="DBSCAN"):
        self.method = method
        _default_distance = 50.0
        self.epsilon=_default_distance/6371.0088
        self.minimum_samples = 3
#        pass

    '''
        TODO consider OPTICS (Ordering Points To Identify the Clustering Structure)
    '''

    '''
        DBSCAN clustering - lat/lon pairs
    '''
    def get_dbscan_labels(self,st_arr, distance_km, minimum_samples):

        import numpy as np
        from sklearn.cluster import DBSCAN
#        from sklearn import metrics
        import sklearn.utils
        from sklearn.preprocessing import StandardScaler
        from sklearn.datasets import make_blobs

        if isinstance(distance_km, float):
            self.epsilon = distance_km/6371.0088
        if isinstance(minimum_samples, int) and minimum_samples > 0:
            self.minimum_samples = minimum_samples

        err="0"
    #    try:
        X, labels_true = make_blobs(n_samples=len(st_arr), centers=st_arr, cluster_std=0.4,random_state=0)
        db = DBSCAN(eps=self.epsilon,
                    min_samples=minimum_samples,
                    algorithm='ball_tree',
                    metric='haversine').fit(np.radians(X))
        print('DBSCAN epsilon:',db.eps,'algorithm:', db.algorithm, 'metric: ', db.metric)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
#        print("DBSCAN found %0.3f labels" % labels )
    #    except Exception as err:
    #        print("Error message:", err)
    #        labels = ""
        return db.labels_, labels_true, core_samples_mask

    '''
        K nearest neigbour clustering
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