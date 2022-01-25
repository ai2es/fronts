import urllib.error
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import requests
import pandas as pd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
matplotlib.use('Qt5Agg')


def plot_surface_background(extent):
    """
    Returns new background for the plot.

    Parameters
    ----------
    extent: Numpy array containing the extent/boundaries of the plot in the format of [min lon, max lon, min lat, max lat].

    Returns
    -------
    ax: New plot background.
    """
    crs = ccrs.LambertConformal(central_longitude=250)
    ax = plt.axes(projection=crs)
    # gridlines = ax.gridlines()
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'), linewidth=0.3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.STATES, linewidth=0.3)
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    return ax


def obtain_buoy_coordinates():
    """ Download buoy observations """
    buoy_stations = pd.read_csv('buoy_stations.csv')
    station_list = list(buoy_stations['ID'])

    for station in station_list:
        url = f'https://www.ndbc.noaa.gov/station_realtime.php?station={station}'

        page = requests.get(url)
        html = str(page.content)

        string_to_find = f'{station} ('
        string_length = len(string_to_find)
        string_location = html.find(string_to_find)
        truncated_html = html[string_location+string_length:string_location+string_length+17]

        try:
            latitude_end = max(truncated_html.find('N'),truncated_html.find('S'))
            latitude_coordinate = float(truncated_html[:latitude_end])

            longitude_end = max(truncated_html.find('E'),truncated_html.find('W'))
            longitude_coordinate = float(truncated_html[latitude_end+2:longitude_end])
        except ValueError:
            print(f"Unable to obtain coordinates for {station}")
        else:
            if 'W' in truncated_html:
                longitude_coordinate = 360-longitude_coordinate

            station_location_in_df = buoy_stations.loc[buoy_stations['ID'] == station].index.to_numpy()[0]

            buoy_stations.at[station_location_in_df, 'lat'] = latitude_coordinate
            buoy_stations.at[station_location_in_df, 'lon'] = longitude_coordinate

            print(f"Successfully obtained coordinates for buoy {station}")

    buoy_stations.to_csv(path_or_buf='buoy_stations.csv')


def plot_buoy_coordinates():
    "Plot buoy coordinates"
    buoy_stations = pd.read_csv('buoy_stations.csv')

    extent = [130,370,0,80]

    fig = plt.figure()
    ax = plot_surface_background(extent)
    buoy_stations.plot.scatter(ax=ax, x='lon', y='lat', s=5, color='red', transform=ccrs.PlateCarree())
    plt.title("Buoy Locations")

    plt.show()


def retrieve_buoy_obs(save_directory):
    "Retrieve buoy observations"
    buoy_station_csv = pd.read_csv('buoy_stations.csv')
    station_list = list(buoy_station_csv['ID'])
    index = 0

    for station in station_list:
        station_info = buoy_station_csv.loc[buoy_station_csv['ID'] == station]
        if station_info['contains_2019'][index]:
            print(f"Retrieving 2019 data for {station} buoy")
            year = 2019
            url = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station}h{year}.txt.gz&dir=data/historical/stdmet/'
            try:
                data = pd.read_csv(url,sep='\s+',low_memory=False)
            except urllib.error.HTTPError:
                print(f"Error retrieving {station} buoy data for 2019 from {url}")
            else:
                data.to_csv(path_or_buf=f"{save_directory}\\{year}\\{station}_{year}.csv")
        if station_info['contains_2020'][index]:
            print(f"Retrieving 2020 data for {station} buoy")
            year = 2020
            url = f'https://www.ndbc.noaa.gov/view_text_file.php?filename={station}h{year}.txt.gz&dir=data/historical/stdmet/'
            try:
                data = pd.read_csv(url,sep='\s+',low_memory=False)
            except urllib.error.HTTPError:
                print(f"Error retrieving {station} buoy data for 2020 from {url}")
            else:
                data.to_csv(path_or_buf=f"{save_directory}\\{year}\\{station}_{year}.csv")
        index += 1


def add_buoy_to_surface_obs(year, month, day, hour, new_directory):
    """ Add buoy obs to surface obs """
    
    timestring = '%d%02d%02d%02d00' % (year, month, day, hour)

    column_names = ['station','valid','lon','lat','tmpf','dwpf','relh','drct','sknt','p01i','alti','mslp','vsby','gust',
                    'skyc1','skyc2','skyc3','skyc4','skyl1','skyl2','skyl3','skyl4','wxcodes','ice_accretion_1hr',
                    'ice_accretion_3hr','ice_accretion_6hr','peak_wind_gust','peak_wind_drct','peak_wind_time',
                    'feel','metar','snowdepth']
    
    surface_obs = pd.read_csv(f"US_sfc_data_{timestring}_{timestring}.txt",skiprows=6, sep=',', low_memory=False,
                     names=column_names, na_values='M')

    buoy_coordinates_csv = pd.read_csv(f"buoy_stations.csv")
    buoy_list = buoy_coordinates_csv[buoy_coordinates_csv[f'contains_{year}'] == True]['ID'].values
    print(buoy_list)
    for buoy in buoy_list:
        try:
            buoy_coordinates_csv = pd.read_csv(f"buoy_stations.csv")
            buoy_obs = pd.read_csv(f"I:\\buoy_obs\\{year}\\{buoy}_{year}.csv",na_values=[99,999,'MM'])

            buoy_info = buoy_coordinates_csv[buoy_coordinates_csv['ID'] == buoy]
            buoy_lat, buoy_lon = buoy_info['lat'].values[0], buoy_info['lon'].values[0]-180

            buoy_obs_month = buoy_obs[buoy_obs['MM'] == '%02d' % month]
            buoy_obs_day = buoy_obs_month[buoy_obs_month['DD'] == '%02d' % day]

            if len(buoy_obs_day) > 0:
                print(f"Adding {buoy} to dataframe")
                for index in list(buoy_obs_day['Unnamed: 0']):
                    buoy_obs_index = buoy_obs_day[buoy_obs_day['Unnamed: 0'] == index]
                    # print(buoy_obs_index)
                    buoy_valid_year, buoy_valid_month, buoy_valid_day, bouy_valid_hour, buoy_valid_min = \
                        buoy_obs_index['#YY'].values, buoy_obs_index['MM'].values, buoy_obs_index['DD'].values, buoy_obs_index['hh'].values, buoy_obs_index['mm'].values
                    buoy_valid_time = '%d-%02d-%02d %02d:%02d:00' % (buoy_valid_year, buoy_valid_month, buoy_valid_day, bouy_valid_hour, buoy_valid_min)
                    buoy_temp_degF = np.array(buoy_obs_index['ATMP'].values[0], dtype=float)*(9/5)+32
                    buoy_dewp_degF = np.array(buoy_obs_index['DEWP'].values[0], dtype=float)*(9/5)+32
                    buoy_pres_hPa = np.array(buoy_obs_index['PRES'].values[0], dtype=float)
                    buoy_wspd_knot = np.array(buoy_obs_index['WSPD'].values[0], dtype=float)*1.94384
                    buoy_wdir_deg = np.array(buoy_obs_index['WDIR'].values[0], dtype=float)

                    buoy_dataframe = pd.DataFrame([[buoy, buoy_valid_time, buoy_lon, buoy_lat, buoy_temp_degF, buoy_dewp_degF, np.NaN,
                        buoy_wdir_deg, buoy_wspd_knot, np.NaN, np.NaN, buoy_pres_hPa, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN,
                        np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]],
                        columns=column_names)

                    surface_obs = surface_obs.append(buoy_dataframe, ignore_index=True)
            else:
                print(f"Error adding {buoy} to dataset")
        except FileNotFoundError:
            print(f"Error: {year} data file for {buoy} buoy not found")

    surface_obs.to_csv(path_or_buf=f"{new_directory}\\MergedObs_{timestring}.csv")


station = '41013'
new_directory = 'I:\\merged_obs'
year, month, day, hours = 2019, 5, 22, [18,21]
for hour in hours:
    add_buoy_to_surface_obs(year, month, day, hour, new_directory)
