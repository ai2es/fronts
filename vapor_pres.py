"""
Function that takes dew-point temperature and calculates the actual vapor pressure, useful in determining pressure centers located along fronts

Written by Colin Willingham (colin.e.willingham-1@ou.edu)

---Sources used in this code ----
(National Weather Service) https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
"""

from dask.distributed import Client, LocalCluster
import xarray
import numpy as np
from math import atan

def vapor_pres(Td):
    """

    Parameters
    ----------
    Td: Dataset
        Xarray dataset containing data for 2-m AGL dew-point temperature. Dew-point temperature has units of kelvin
        (K).

    Returns
    -------
    e: Dataset
        Xarray dataset containing data for saturated vapor pressure

    """

    e= 6.11*np.power(10, (7.5*Td)/(237.3+Td))

    return e

if  __name__ == "__main__":
    #cluster = LocalCluster()
    #client = Client(cluster)
    e=vapor_pres(30)
    print(e)
