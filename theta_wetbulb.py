#Optional - May Not work on your system.
from dask.distributed import Client, LocalCluster

cluster = LocalCluster()
client = Client(cluster)
cluster

"""
Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 3/6/2021 11:16 CST
Modified for Xarray by: John Allen (allen4jt@cmich.edu)
"""

import xarray as xr
import numpy as np
import argparse

def theta_w_calculation(T, Td, P):
    # constants
    L0 = 2.5*(10**6) # Latent heat of vaporization of water (J/kg).
    L1 = 2.37*(10**3) # constant from paper (J/kg/K)
    Rv = 461.5 # Gas constant for water vapor (J/kg/K)
    Rd = 287.04 # Gas constant for dry air (J/kg/K)
    epsilon = 0.622 # Rd/Rv
    cpd = 1005 # Specific heat of dry air (J/kg/K)
    c = 4186 # Specific heat of water (J/kg/K)
    e_knot = 6.11 # hPa
    T_knot = 273.15 # K
    C = 273.15 # K
    p_knot = 1000 # hPa
    exp = 2.71828 # e constant, declared as exp to avoid confusion with vapor pressure variable e
    kd = 0.286 # Rd/cpd
    gamma = 3.504  # 1/kd

    P = P/100.
    e = e_knot*(np.exp((L0/Rv)*((1/T_knot)-(1/Td))))
    #print(e.max().values)
    es = e_knot*(np.exp((L0/Rv)*((1/T_knot)-(1/T))))
    #print(es.max().values)

    w = (epsilon*e)/(P-e) # gH2O/g(air)
    #print(w.max().values)
    ws = epsilon*es/(P-es) # saturation mixing ratio
    #print(es.max().values)

    TL = (((1/(Td-56))+(np.log(T/Td)/800))**(-1))+56
    #print(TL.max().values)

    # LCL potential temperature
    theta_L = T*((p_knot/(P-e))**kd)*((T/TL)**(0.28*w))
    #print(theta_L.max().values)
    #print("theta_L = %.01f°C" % round(theta_L-273.15, 1))

    # equivalent potential temperature
    theta_e = theta_L*(np.exp(((3036/TL)-1.78)*w*(1+(0.448*w))))
    #print(theta_e.max().values)

    # wet-bulb potential temperature
    a0 = 7.101574
    a1 = -20.68208
    a2 = 16.11182
    a3 = 2.574631
    a4 = -5.205688
    b1 = -3.552497
    b2 = 3.781782
    b3 = -0.6899655
    b4 = -0.592934
    X = theta_e/C

    #theta_wc = theta_e - (np.exp((a0+(a1*X)+(a2*X**2)+(a3*X**3)+(a4*X**4))/(1+(b1*X)+(b2*X**2)+(b3*X**3)+(b4*X**4))))
    theta_wc = theta_e - (np.exp((a0+(a1*X)+(a2*np.power(X,2))+(a3*np.power(X,3))+(a4*np.power(X,4)))/(1+(b1*X)+(b2*np.power(X,2))+(b3*np.power(X,3))+(b4*np.power(X,4)))))
    
    theta_w = xr.where(theta_e > 173.15, theta_wc, theta_e).compute()
    return theta_w
    
#Testing
chunks={'time': 1}
data = xr.open_dataset('ERA5_2018_3hrly_sp.nc',chunks=chunks)
data1 = xr.open_dataset('ERA5_2018_3hrly_2mT.nc',chunks=chunks)
data2 = xr.open_dataset('ERA5_2018_3hrly_2mTd.nc',chunks=chunks)

ds = xr.merge([data1, data2, data])

from timeit import default_timer as timer
start = timer()
theta_w =  theta_w_calculation(ds.t2m, ds.d2m, ds.sp)
end = timer()
print('Theta-W 1 Year ',end-start)
