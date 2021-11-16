"""
Example script that scrapes data from the IEM ASOS download service
"""
from __future__ import print_function
import json
import time
import datetime
# Python 2 and 3: alternative 4
try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

# Number of attempts to download data
MAX_ATTEMPTS = 6
# HTTPS here can be problematic for installs that don't have Lets Encrypt CA
SERVICE = "http://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?"


def download_data(uri):
    """Fetch the data from the IEM
    The IEM download service has some protections in place to keep the number
    of inbound requests in check.  This function implements an exponential
    backoff to keep individual downloads from erroring.
    Args:
      uri (string): URL to fetch
    Returns:
      string data
    """
    attempt = 0
    while attempt < MAX_ATTEMPTS:
        try:
            data = urlopen(uri, timeout=300).read().decode('utf-8')
            if data is not None and not data.startswith('ERROR'):
                return data
        except Exception as exp:
            print("download_data(%s) failed with %s" % (uri, exp))
            time.sleep(5)
        attempt += 1

    print("Exhausted attempts to download, returning empty data")
    return ""


def get_stations_from_filelist(filename):
    """Build a listing of stations from a simple file listing the stations.
    The file should simply have one station per line.
    """
    stations = []
    for line in open(filename):
        stations.append(line.strip())
    return stations


def get_stations_from_networks():
    """Build a station list by using a bunch of IEM networks."""
    stations = []
    states = """AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME
     MI MN MO MS MT NC ND NE NH NJ NM NV NY OH OK OR PA RI SC SD TN TX UT VA VT
     WA WI WV WY CA_BC CA_AB CA_SK CA_MB CA_ON CA_QC CA_NS CA_NF CA_NB CA_NT
     CA_NU CA_PE CA_YT"""
    # IEM quirk to have Iowa AWOS sites in its own labeled network
    networks = ['AWOS']
    for state in states.split():
        networks.append("%s_ASOS" % (state,))

    for network in networks:
        # Get metadata
        uri = ("https://mesonet.agron.iastate.edu/"
               "geojson/network/%s.geojson") % (network,)
        data = urlopen(uri)
        jdict = json.load(data)
        for site in jdict['features']:
            stations.append(site['properties']['sid'])
    return stations


def main():
    """Our main method"""
    # timestamps in UTC to request data for
    # Change these dates for what you need. Typically one day of
    # data will amount to ~24 mb
    startts = datetime.datetime(2019, 5, 22, 21)
    endts = datetime.datetime(2019, 5, 22, 21)

    service = SERVICE + "data=all&tz=Etc/UTC&format=comma&latlon=yes&"

    service += startts.strftime('year1=%Y&month1=%m&day1=%d&')
    service += endts.strftime('year2=%Y&month2=%m&day2=%d&')

    # Two examples of how to specify a list of stations
    stations = get_stations_from_networks()
    nstns = len(stations)
    # stations = get_stations_from_filelist("mystations.txt")

    outfn = '{}_{}_{}.txt'.format('US_sfc_data', startts.strftime("%Y%m%d%H%M"),
                                  endts.strftime("%Y%m%d%H%M"))
    out = open(outfn, 'w')

    # Subset number of stations to download to not create too long
    # of a search string.

    # The step of 500 has been chosen as a value that will allow the
    # greatest number of stations to be downloaded at one time without
    # causing too large of a search string
    step = 500
    i = 1
    for part in range(0, nstns, step):
        uri = service
        endpart = part+step
        if endpart > nstns:
            endpart = nstns
        for station in stations[part:endpart]:
            uri += '&station={}'.format(station)
        print('Downloading: US Part {}'.format(i))
        data = download_data(uri)
        # This if statment will get rid of the "header" infomation
        # by starting to write to the file after it. This allows for
        # a relatively easy method to create a single file for the
        # data being downloaded in pieces
        if i > 1:
            head_end = data.find('metar')
            out.write(data[head_end+6:])
        else:
            out.write(data)
        i += 1
    out.close()

if __name__ == '__main__':
    main()
