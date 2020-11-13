# Imports
import numpy as np
import xarray as xr
import matplotlib as mpl

mpl.use('agg', force=True)
import matplotlib.pyplot as plt

plt.switch_backend('agg')
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata
import xml.etree.ElementTree as ET
import pandas as pd
import glob
import argparse
import os


def read_xml_files(year, month, day):
    """
    Read in xml files for climatology
    :param year:
    :param month:
    :param day:
    :return:
    """
    file_path = "/home/awoodward/xmls_2006122012-2020061900/%04d" % (year)
    print(file_path)
    # read all the files in the directory
    files = glob.glob("%s/*%04d%02d%02d*.xml" % (file_path, year, month, day))
    dss = []
    for filename in files:
        print(filename)
        # Load XML Data
        tree = ET.parse(filename)
        root = tree.getroot()
        # Split Off Date, Could be Useful later
        date = filename.split('_')[-1].split('.')[0].split('f')[0]

        # Define Holding Lists for Data as Read from XML
        fronttype = []
        frontnumber = []
        lats = []
        lons = []
        dates = []
        # Define counter for front number.
        i = 0
        # Iterate through all Frontal Line objects
        for line in root.iter('Line'):
            # Add to counter for each new front.
            i = i + 1
            # Store relevant info for dataframe.
            frontno = i
            frontty = line.get("pgenType")
            for point in line.iter('Point'):
                dates.append(date)
                frontnumber.append(frontno)
                fronttype.append(frontty)
                lats.append(point.get("Lat"))
                lons.append(point.get("Lon"))
                #                print(lats,lons)
        # Now Create a Dataframe of the lists using the zip approach.
        df = pd.DataFrame(list(zip(dates, frontnumber, fronttype, lats, lons)),
                          columns=['Date', 'Front Number', 'Front Type', 'Latitude', 'Longitude'])
        # Define Types to Avoid trouble later
        df['Latitude'] = df.Latitude.astype(float)
        df['Longitude'] = df.Longitude.astype(float)
        df['Front Number'] = df['Front Number'].astype(int)
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d%H')

        # Define your grid - here I am using 0.25 degree
        # xit = np.linspace(130,370,961)
        xit = np.concatenate((np.linspace(-180, 10, 761), np.linspace(170, 180, 41)[0:40]))
        yit = np.linspace(0, 80, 321)

        # Now Use a Little Trick to Map the data to a New Grid as Pandas Data Columns - in Place.
        # Note that if the values are outside the bins, then these will be placed in 0, len(xit).
        # Remember there is always one less bin than there is a value.
        df = df.assign(xit=np.digitize(df['Longitude'].values, xit))
        df = df.assign(yit=np.digitize(df['Latitude'].values, yit))

        types = df['Front Type'].unique()

        type_da = []
        for i in range(0, len(types)):
            type_df = df[df['Front Type'] == types[i]]
            groups = type_df.groupby(['xit', 'yit'])

            # Create a Numpy Array to Count Frequency
            frequency = np.zeros((yit.shape[0] + 1, xit.shape[0] + 1))

            for group in groups:
                # Here we count in the appropriate location, and avoid double counting - assign to location
                frequency[group[1].yit.values[0], group[1].xit.values[0]] += np.where(
                    len(group[1]['Front Number']) >= 1, 1, 0)

                # Trim to clean up the final bin where anything outside domain lands
            frequency = frequency[1:322, 1:802]

            # Now turn it into an xarray DataArray
            ds = xr.Dataset(data_vars={"Frequency": (['Latitude', 'Longitude'], frequency)},
                            coords={'Latitude': yit, 'Longitude': xit})
            # ds = xr.DataArray(frequency,dims=('Latitude', 'Longitude'),coords={'Latitude': yit,'Longitude':xit})
            # Append this type of front to the list for concatenation
            type_da.append(ds)
        # Concatenate Along a New Dimension we will label 'Front Type'
        ds = xr.concat(type_da, dim=types)
        ds = ds.rename({'concat_dim': 'Type'})
        dss.append(ds)
    timestep = pd.date_range(start=df['Date'].dt.strftime('%Y-%m-%d')[0], periods=8, freq='3H')
    dns = xr.concat(dss, dim=timestep)
    dns = dns.rename({'concat_dim': 'Date'})
    return dns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--year', type=int, required=True, help="year for the data to be read in")
    parser.add_argument('--month', type=int, required=True, help="month for the data to be read in")
    parser.add_argument('--day', type=int, required=True, help="day for the data to be read in")
    #   parser.add_argument('--image_outdir', type=str, required=False, help="output directory for image files")
    parser.add_argument('--netcdf_outdir', type=str, required=False, help="output directory for netcdf files")
    args = parser.parse_args()
    #   print(args)
    # read the polygons for the specified day
    #   print("reading the xml")
    xmls = read_xml_files(args.year, args.month, args.day)
    #    print(xmls)

    # One way to output - As a netCDF of just this timestep - the above could then be looped in bash,
    # and xarray used to open the multiple nc files:
    netcdf_outtime=str('%04d%02d%02d'%(args.year, args.month, args.day))
    xmls.to_netcdf(os.path.join([args.netcdf_outdir, f'FrontalCounts_{netcdf_outtime}.nc']))

# Here is an idea of how you might loop through your list of files to make this have a time dimension as well, using a list of xarrays
# which you then concatenate.
# dss = []
# ds = xr.DataArray(CF_frequency,dims=('Latitude', 'Longitude'),coords={'Latitude': yit,'Longitude':xit})
# dss.append(ds)
# ds = xr.concat(dss, dim=dates, name='Time')

# cartopy map
#   for i in range(0,8):
#       fig = plt.figure(figsize=(10, 8))
#       ax = plt.axes(projection=ccrs.LambertConformal(central_longitude=250))
#       ax.gridlines()
#       ax.add_feature(cfeature.COASTLINE)
#       ax.add_feature(cfeature.BORDERS)
#       ax.add_feature(cfeature.STATES)
#       ax.set_extent([130, 370, 0, 80], crs=ccrs.PlateCarree())

# Ryan function to increase number of points in fronts


# plot
#       xmls.Frequency.sel(Type= 'COLD_FRONT').isel(Date=i).plot(x='Longitude',y='Latitude',transform=ccrs.PlateCarree())
#       xmls.Frequency.sel(Type= 'WARM_FRONT').isel(Date=i).plot(x='Longitude',y='Latitude',transform=ccrs.PlateCarree())
#       outtime=str(xmls.Frequency.Date[i].dt.strftime('%Y%m%d%H').values)
#       print(outtime)
#       plt.savefig(os.path.join(args.image_outdir,f'frequencyplot_{outtime}.png'), bbox_inches='tight',dpi=300)
# plt.show()
