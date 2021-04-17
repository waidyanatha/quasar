#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''

CLASS for defining station, channel, fault types and coding and data filter functions

We make use of the International Federation Data of Seismic Networks (FDSN), the global standard
and a [data service](http://www.fdsn.org/services/) for sharing seismic sensor wave form data.
The Obspy librarires support FDSN. The list of resources and services that are used for retrieving station inventory and
waveform data.

* FSDN as Client data sources; both (i) the FDSN client service and the (ii) FDSN complient GoeNet API webservice
* FDSN station service - retrieve station metadata information in a FDSN StationXML format or text format for all
the channels in CECS station with no time limitations:
https://service.geonet.org.nz/fdsnws/station/1/query?network=NZ&station=CECS&level=channel&format=text

Prepare an array of tuples necessary and sufficient station data:
* _station code_ as a unique identifier
* _coordinates_ longitude & latitude
* _elevation_ in meters above mean sea level

'''

from obspy import read_inventory, UTCDateTime

''' All weak & strong motion, low gain, and mass possion sensor types '''
class station_data():

    def __init__(self, name: str = 'station_metadata'):

        '''
            Define the data source to make the connection and set the local variables
        '''

        self.name = name

        ''' Establish start and end time for retrieving waveform data '''
        self.t_start = UTCDateTime.now()-518400 #6 days ago = 60s x 60m x 24h x 6d
        self.t_end = UTCDateTime.now()+86400 #1 day in the future = 60s x 60m x 24h

        st_test_data = ""     # TODO create a test set with corresponding faults
        ''' Use either or GeoNet station service webservice URL or Obspy FDSN Client protocol to retrieve station data '''
        self.s_fdsn_url_code = "GEONET"     # FDSN client URL code

        #st_ws = 'https://service.geonet.org.nz/fdsnws/station/1/query?network=NZ&station=CECS&level=channel'
        st_ws = 'https://service.geonet.org.nz/fdsnws/station/1/query?network=NZ&level=station&endafter=2020-12-31&format=xml'
        ''' NZ faults '''
#        s_flt_full_data = "../data/NZAFD/JSON/NZAFD_Oct_2020_WGS84.json"  # Downloaded and unzipped data
#        s_flt_test_data = "../data/NZAFD/JSON/NZAFD_WGS84-test.json"     # Sample of 6-10 fault attribs & features
#        s_flt_new_data = "https://data.gns.cri.nz/af/"                # Active Faults GeoNet database

#        return name

    def get_client(self):

        from obspy.clients.fdsn import Client

        try:
            client  = Client(self.s_fdsn_url_code)
#            print(client)

        except Exception as err:
            print("Error message: [get_client]", err)

        return client

    ''' Define channel codes '''
    def get_channels(self):

        channels = "UH*,VH*,LH*,BH*,SH*,HH*,EH*,UN*,VN*,LN*,BN*,SN*,HN*,EN*"

        return channels

    ''' All combinations with definition of the first and second letter to define identify each station type '''
    def get_types(self):
        dict_st_types = {"UH" : "Weak motion sensor,\ne.g., measuring velocity\nUltra Long Period sampled\nat 0.01Hz, or SOH sampled at 0.01Hz",
                   "VH" : "Weak motion sensor,\ne.g., measuring velocity\nVery Long Period sampled\nat 0.1Hz, or SOH sampled at 0.1Hz",
                   "LH" : "Weak motion sensor,\ne.g., measuring velocity\nBroad band sampled\nat 1Hz, or SOH sampled at 1Hz",
                   "BH" : "Weak motion sensor,\ne.g., measuring velocity\nBroad band sampled\nat between 10 and 80 Hz, usually 10 or 50 Hz",
                   "SH" : "Weak motion sensor,\ne.g., measuring velocity\nShort-period sampled\nat between 10 and 80 Hz, usually 50 Hz",
                   "HH" : "Weak motion sensor,\ne.g., measuring velocity\nHigh Broad band sampled\nat or above 80Hz,\ngenerally 100 or 200 Hz",
                   "EH" : "Weak motion sensor,\ne.g., measuring velocity\nExtremely Short-period sampled\nat or above 80Hz, generally 100 Hz",
                   "UN" : "Strong motion sensor,\ne.g., measuring acceleration\nUltra Long Period sampled\nat 0.01Hz, or SOH sampled at 0.01Hz",
                   "VN" : "Strong motion sensor,\ne.g., measuring acceleration\nVery Long Period sampled\nat 0.1Hz, or SOH sampled at 0.1Hz",
                   "LN" : "Strong motion sensor,\ne.g., measuring acceleration\nBroad band sampled\nat 1Hz, or SOH sampled at 1Hz",
                   "BN" : "Strong motion sensor, e.g. measuring acceleration\nBroad band sampled\nat between 10 and 80 Hz, usually 10 or 50 Hz",
                   "SN" : "Strong motion sensor,\ne.g., measuring acceleration\nShort-period sampled\nat between 10 and 80 Hz, usually 50 Hz",
                   "HN" : "Strong motion sensor,\ne.g., measuring acceleration\nHigh Broad band sampled\nat or above 80Hz,\ngenerally 100 or 200 Hz",
                   "EN" : "Strong motion sensor,\ne.g., measuring acceleration\nExtremely Short-period sampled\nat or above 80Hz, generally 100 Hz"}
        return dict_st_types

    ''' TODO Ranking of the station types by their EEW capacity and capabilities
        currently simply enumerating them for testing '''
    def get_st_type_rank(self):

        l_enum_st_types = []
        try:
            for idx_st_type, val_st_type in enumerate(list(self.get_types())):
                l_enum_st_types.append([idx_st_type, val_st_type])
        except Exception as err:
            print("Error message: [get_st_type_rank]", err)
            sys.exit(1)
        return l_enum_st_types

    '''
        Prepare an array of station data: (i) station code as a unique identifier,
                                        (ii) coordinates longitude & latitude, and
                                        (iii) elevation in meters above mean sea level
        return the construct as a list of stations including the list of invalid stations
    '''
    def get_stations(self, client):
        st_list = []
        invalid_st_list = []

        try:
            st_inv = client.get_stations(network='NZ', location="1?,2?", station='*', channel=self.get_channels(), level='channel', starttime=self.t_start, endtime = self.t_end)
        except Exception as err:
            print("Error message: [get_stations]", err)

        '''run through stations to parse code, type, and location'''
        try:
            for each_st in range(len(st_inv[0].stations)):
                ''' use lat/lon paris only in and around NZ remove all others '''
                if(st_inv[0].stations[each_st].latitude < 0 and st_inv[0].stations[each_st].longitude > 0):
                    each_st_type_dict = st_inv[0].stations[each_st].get_contents()
                    ''' get the second character representing the station type '''
#                    st_type_dict["st_type"].append(each_st_type_dict["channels"][0][-3:-1])
                    ''' list of corresponding station locations (lat / lon) '''
                    st_list.append([st_inv[0].stations[each_st].code, each_st_type_dict["channels"][0][-3:-1], st_inv[0][each_st].latitude, st_inv[0][each_st].longitude])
                else:
                    ''' dictionary of all stations not in NZ visinity '''
                    invalid_st_list.append([st_inv[0].stations[each_st].code,st_inv[0][each_st].latitude, st_inv[0][each_st].longitude])

        except Exception as err:
            print("Error message: [get_stations]", err)

        return st_list, invalid_st_list, st_inv

    ''' DEPRECATE -- GET WAVE FORMS '''
    def get_station_waveform(self, client, station_code, **kwargs):

        StartTime = (kwargs["StartTime"] if kwargs["StartTime"]!=None else self.t_start)
        EndTime = (kwargs["EndTime"] if kwargs["EndTime"]!=None else self.t_end)

#        st_wf = client.get_waveforms("NZ", station_code,"*", "H??", self.t_start, self.t_end, attach_response=True)
#        st_wf = client.get_waveforms("NZ", station_code,"*", "H??", StartTime, EndTime, attach_response=True)
#        return st_wf
        return client.get_waveforms("NZ", station_code,"*", "H??", StartTime, EndTime, attach_response=True)