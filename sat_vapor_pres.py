"""
Function that takes air temperature and calculates the actual saturated vapor pressure, useful in determining pressure centers located along fronts

Written by Colin Willingham (colin.e.willingham-1@ou.edu)

---Sources used in this code ----
(National Weather Service) https://www.weather.gov/media/epz/wxcalc/vaporPressure.pdf
"""

from dask.distributed import Client, LocalCluster
import xarray
import numpy as np

def sat_vapor_pres_calculation(T):
    """

    Parameters
    ----------
    T: Dataset
        Xarray dataset containing data for 2-m AGL air temperature. Dew-point temperature has units of kelvin
        (K).

    Returns
    -------
    sat_vapor_pres: Dataset
        Xarray dataset containing data for saturated vapor pressure

    """

es= 6.11*10**((7.5*T)/(237.3+T))

return sat_vapor_pres

__name__ == "__main__":
    cluster = LocalCluster()
    client = Client(cluster)
