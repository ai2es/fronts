# Wet-bulb potential temperature (theta_w)

"""
Code written by: Andrew Justin (andrewjustin@ou.edu)
Last updated: 3/6/2021 11:16 CST
"""

import xarray as xr
import numpy as np
import argparse
import math

# wet-bulb temperature calculation
def theta_w_calculation(ds_2mT, ds_2mTd, ds_sp):

    # constants
    L0 = 2.5*(10**6) # Latent heat of vaporization of water (J/kg).
    L1 = 2.37*(10**3) # constant from paper (J/kg/K)
    Rv = 461.5 # Gas constant for water vapor (J/kg/K)
    Rd = 287 # Gas constant for dry air (J/kg/K)
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

    theta_w_array = np.array([[0.0000000]*len(ds_2mT.t2m[0].values)]*len(ds_2mT.t2m.values))

    lat = ds_2mT['latitude'].values
    lon = ds_2mT['longitude'].values

    for a in range(0,len(ds_2mT.t2m.values)):
        print("Column %d/317" % (a+1))
        for b in range(0,len(ds_2mT.t2m[a].values)):
            T = ds_2mT.t2m[a][b].values
            Td = ds_2mTd.d2m[a][b].values
            p = ds_sp.sp[a][b].values/100 # convert pressure from Pa to hPa
            #print("T = %f K" % T)
            # vapor pressure
            e = e_knot*(exp**((L0/Rv)*((1/T_knot)-(1/Td))))
            es = e_knot*(exp**((L0/Rv)*((1/T_knot)-(1/T))))
            #print("%f hPa" % e)

            # mixing ratios
            w = (epsilon*e)/(p-e) # gH2O/g(air)
            ws = epsilon*es/(p-es) # saturation mixing ratio
            #print("%f g/g" % w)

            # LCL temperature
            TL = (((1/(Td-56))+(math.log(T/Td)/800))**(-1))+56
            #print("TL = %.01f°C" % round(TL-273.15, 1))

            # LCL potential temperature
            theta_L = T*((p_knot/(p-e))**kd)*((T/TL)**(0.28*w))
            #print("theta_L = %.01f°C" % round(theta_L-273.15, 1))

            # equivalent potential temperature
            theta_e = theta_L*(exp**(((3036/TL)-1.78)*w*(1+(0.448*w))))
            #print("theta_e = %.01f°C" % round(theta_e-273.15, 1))

            # wet-bulb potential temperature
            if theta_e > 173.15:
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

                theta_w = theta_e - (exp**((a0+(a1*X)+(a2*X**2)+(a3*X**3)+(a4*X**4))/(1+(b1*X)+(b2*X**2)+(b3*X**3)+(b4*X**4))))
                theta_w_array[a][b] = theta_w

            else:
                theta_w = theta_e
                theta_w_array[a][b] = theta_w

    da_theta_w = xr.DataArray(name='theta_w', data=theta_w_array, coords={'latitude': lat,'longitude': lon}, dims=['latitude','longitude'])

    ds_theta_w = da_theta_w.to_dataset()

    return ds_theta_w

def theta_w_test(T, Td, p):

    T = T + 273.15
    Td = Td + 273.15

    # constants
    L0 = 2.5*(10**6) # Latent heat of vaporization of water (J/kg).
    L1 = 2.37*(10**3) # constant from paper (J/kg/K)
    Rv = 461.5 # Gas constant for water vapor (J/kg/K)
    Rd = 287 # Gas constant for dry air (J/kg/K)
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

    # vapor pressure
    e = e_knot*(exp**((L0/Rv)*((1/T_knot)-(1/Td))))
    es = e_knot*(exp**((L0/Rv)*((1/T_knot)-(1/T))))
    print("%f hPa" % e)

    # mixing ratios
    w = (epsilon*e)/(p-e) # gH2O/g(air)
    ws = epsilon*es/(p-es) # saturation mixing ratio
    print("%f g/g" % w)

    # LCL temperature
    TL = (((1/(Td-56))+(math.log(T/Td)/800))**(-1))+56
    print("TL = %.01f°C" % round(TL-273.15, 1))

    # LCL potential temperature
    theta_L = T*((p_knot/(p-e))**kd)*((T/TL)**(0.28*w))
    print("theta_L = %.01f°C" % round(theta_L-273.15, 1))

    # equivalent potential temperature
    theta_e = theta_L*(exp**(((3036/TL)-1.78)*w*(1+(0.448*w))))
    print("theta_e = %.01f°C" % round(theta_e-273.15, 1))

    # wet-bulb potential temperature
    if theta_e > 173.15:
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

        theta_w = theta_e - C - (exp**((a0+(a1*X)+(a2*X**2)+(a3*X**3)+(a4*X**4))/(1+(b1*X)+(b2*X**2)+(b3*X**3)+(b4*X**4))))

    else:
        theta_w = theta_e - C

    print("theta_w = %.01f°C" % round(theta_w, 1))

    return theta_w

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--T', type=float, required=False, help="temperature in °C")
    parser.add_argument('--Td', type=float, required=False, help="dew point temperature in °C")
    parser.add_argument('--p', type=float, required=False, help="pressure in hPa")
    args = parser.parse_args()

    theta_w_test(args.T, args.Td, args.p)
