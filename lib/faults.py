#!/usr/bin/env python
# coding: utf-8

# In[1]:


'''
    CLASS of functions for offering various fault-line data filters, clensing, and structuring procedures
'''
class fault_data():

    '''
        TODO at initiatlization download latest ZIP'd datasets from GeoNet then extract the *.json
    '''
    def __init__(self, name: str = 'Fault Metadata'):

        self.name = name

        ''' NZ fault datasets '''
        self.s_flt_full_data = "../data/NZAFD/JSON/NZAFD_Oct_2020_WGS84.json"  # Downloaded and unzipped data
        self.s_flt_test_data = "../data/NZAFD/JSON/NZAFD_WGS84-test.json"     # Sample of 6-10 fault attribs & features
        self.s_flt_new_data = "https://data.gns.cri.nz/af/"                # Active Faults GeoNet database

#        pass

    ''' Return the fault file name '''
    def get_fault_file(self, s_type: str = 'test'):
        return self.s_flt_test_data

    ''' Extract nested values from a JSON tree to build a list of fault lines
        containing the fault name and lat / lon pairs of the path '''

    def get_paths(self, s_file: str = None):

        import json
        from dictor import dictor

        self.s_file = self.s_flt_test_data
        print(self.s_file)
        try:
#            with open('../data/NZAFD/JSON/NZAFD_Oct_2020_WGS84.json') as json_file:
#                data = json.load(json_file)
#            with open('../data/NZAFD/JSON/NZAFD_WGS84-test.json') as json_file:
#                data = json.load(json_file)
            ''' change parameter to switch between test, full downloaded, and latest data sets
                test: s_flt_test_data
                full: s_flt_full_data
                new: s_flt_new_data '''
            with open(s_file) as json_file:
                data = json.load(json_file)

            faults = []
            fault_path_count = 1
            for each_feature in range(len(data['features'])):
                s_flt_id = dictor(data,('features.{0}.attributes.FID').format(each_feature))
                s_flt_name = dictor(data,('features.{0}.attributes.NAME').format(each_feature))
                s_flt_uid = str(s_flt_id) + " " + s_flt_name
                if s_flt_uid==" ":
                    s_flt_uid = 'Unnamed fault '+ str(fault_path_count)
                    fault_path_count += 1
                points = []
                path = dictor(data,'features.{}.geometry.paths.0'.format(each_feature))
                for each_coordinate in range(len(path)):
                    points.append([path[each_coordinate][0],path[each_coordinate][1]])
                faults.append([s_flt_uid,points])

            '''
            faults = []
            fault_path_count = 1
            for each_feature in range(len(data['features'])):
                flt = dictor(data,('features.{}.attributes.FID'+' '+'features.{}.attributes.NAME').format(each_feature))
                if flt==" ":
                    flt = 'Unnamed fault '+ str(fault_path_count)
                    fault_path_count += 1
                points = []
                path = dictor(data,'features.{}.geometry.paths.0'.format(each_feature))
                for each_coordinate in range(len(path)):
                    points.append([path[each_coordinate][0],path[each_coordinate][1]])
                faults.append([flt,points])
            '''
        except Exception as err:
            print("Error message:", err)
        return faults

    '''
        Interpolate more points for each fault line; if the distance between points > 1.5Km @ 0.5Km intervals
        Otherwise, fit a single halfway point
    '''
    def interpolate_paths(self, paths, distance=float(2.5)):
        from shapely.geometry import LineString

        interp_paths = []
        try:
            ''' loop through each fault path to breakdown into line segments; i.e. coordinate pairs '''
            for path in range(len(paths)):
                path_index = 0
                ''' add the two line segment coordinates to begin with
                    now loop through each path line segment to add interpolated points  '''
                while (path_index < len(paths[path][1])-1):
                    ip = []     # interpolated point
                    rel_origin_coord = paths[path][1][path_index]     # relative starting point of the path
                    rel_nn_coord = paths[path][1][path_index+1]

                    ''' change to a while loop until all distances between consecutive points < delta_distance'''
                    while LineString([rel_origin_coord, rel_nn_coord]).length*6371.0 > distance:
                        ip = LineString([rel_origin_coord,rel_nn_coord]).interpolate((10.0**3)/6371.0, normalized=True).wkt
                        # convertion needs to happen otherwise throws an exception
                        ip_lat = float(ip[ip.find("(")+1:ip.find(")")].split()[0])
                        ip_lon = float(ip[ip.find("(")+1:ip.find(")")].split()[1])
                        rel_nn_coord = list([ip_lat,ip_lon])
                        ''' If you want to add the already interpolated coordinates to the path to possibly speedup
                        and use those points to create a denser path; note that it may will results in uniequal
                        distant between consecutive points in the path. Comment the instruction below to disable.
                        '''
                        paths[path][1].insert(path_index+1,rel_nn_coord)    # interpolated coordinates closest to the relative origin

                    path_index += 1

                interp_paths.append([paths[path][0], paths[path][1]])

        except Exception as err:
            print("Error message:", err)

        return interp_paths