"""
Function that takes air temperature, dew-point temperature, and pressure datasets and returns a new dataset containing
wet-bulb potential temperature (theta-w) with the same coordinates.

Code written by: Andrew Justin (andrewjustin@ou.edu)
Modified for Xarray by: John Allen (allen4jt@cmich.edu)
Last updated: 3/9/2021 11:29 CST by Andrew Justin

---Sources used in this code---
(Bolton 1980): https://doi.org/10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
(Davies-Jones 2008): https://doi.org/10.1175/2007MWR2224.1
"""

from dask.distributed import Client, LocalCluster
import xarray as xr
import numpy as np

def theta_w_calculation(T, Td, P):
    """
    Returns dataset containing wet-bulb potential temperature (theta-w) data with units of kelvin (K).

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for 2-meter AGL air temperature. Air temperature has units of kelvin (K).
    Td: Dataset
        Xarray dataset containing data for 2-meter AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).
    P: Dataset
        Xarray dataset containing data for surface pressure. Surface pressure has units of pascals (Pa).

    Returns
    -------
    theta_w: Dataset
        Xarray dataset containing data for 2-meter AGL wet-bulb potential temperature. Wet-bulb potential temperature
        has units of kelvin (K).

    """

    L = 2.5*(10**6) # Latent heat of vaporization of water (J/kg).
    Rv = 461.5 # Gas constant for water vapor (J/kg/K).
    epsilon = 0.622
    e_knot = 6.11 # hPa
    T_knot = 273.15 # K
    C = 273.15 # K (Davies-Jones 2008, Section 2)
    p_knot = 1000 # hPa
    kd = 0.286

    P = P/100 # Convert surface pressure from Pa to hPa (hectopascals).

    e = e_knot*(np.exp((L/Rv)*((1/T_knot)-(1/Td))))
    w = (epsilon*e)/(P-e)
    TL = (((1/(Td-56))+(np.log(T/Td)/800))**(-1))+56 # LCL temperature (Bolton 1980, Eq. 15)
    theta_L = T*((p_knot/(P-e))**kd)*((T/TL)**(0.28*w)) # LCL potential temperature (Bolton 1980, Eq. 24)
    theta_e = theta_L*(np.exp(((3036/TL)-1.78)*w*(1+(0.448*w)))) # (Bolton 1980, Eq. 39)

    # Wet-bulb potential temperature constants and X variable (Davies-Jones 2008, Section 3).
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

    # Wet-bulb potential temperature approximation (Davies-Jones 2008, Eq. 3.8).
    theta_wc = theta_e - np.exp((a0+(a1*X)+(a2*np.power(X,2))+(a3*np.power(X,3))+(a4*np.power(X,4)))/
                                 (1+(b1*X)+(b2*np.power(X,2))+(b3*np.power(X,3))+(b4*np.power(X,4))))
    theta_w = xr.where(theta_e > 173.15, theta_wc, theta_e)

    return theta_w

if __name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
