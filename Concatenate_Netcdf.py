# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 23:07:47 2020

@author: AlyssaWoodward
"""

import xarray as xr
import netCDF4
import glob
import numpy

#for loop
#for filename in glob.glob('C:/Users/AlyssaWoodward/Documents/Grad School/Research/*.nc'):
#    print("merging")
#    dsmerged = xr.merge([filename])

file_path='/home/awoodward/xmls_2006122012-2020061900/output_netcdf/'
folder='2007/'
#list comprehension
ds = xr.merge([xr.open_dataset(f) for f in glob.glob(file_path+folder+'*.nc')])
xr.open_mfdataset()

ds = xr.open_mfdataset('FrontalCounts_2007*.nc', combine='by_coords', concat_dim="time")
ds.to_netcdf('2007_combined.nc')

#John's code
#dss = []
#ds = xr.DataArray(CF_frequency,dims=('Latitude', 'Longitude'),coords={'Latitude': yit,'Longitude':xit})
#dss.append(ds)
#ds = xr.concat(dss, dim=dates, name='Time')