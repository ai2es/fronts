"""
Numerical frontal analysis (NFA) methods.

References
----------
* Renard and Clarke 1965: https://doi.org/10.1175/1520-0493(1965)093%3C0547:EINOFA%3E2.3.CO;2
* Clarke and Renard 1966: https://doi.org/10.1175/1520-0450(1966)005%3C0764:TUSNNF%3E2.0.CO;2

Author: Andrew Justin (andrewjustinwx@gmail.com)
Script version: 2023.10.6
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # this line allows us to import scripts outside the current directory
import numpy as np
from utils import data_utils


def thermal_front_parameter(field, lats, lons):
    """
    Calculates the thermal front parameter (TFP; Renard and Clarke 1965) using a provided thermodynamic variable.

    Parameter
    ---------
    field: array-like of shape (..., M, N)
        2-D field of the thermodynamic variable that will be used to calculate the TFP. The last two axes of length M and
            N must be the latitude and longitude axes, respectively.
    lats: array-like of shape (M,)
        1-D vector of latitude points expressed as degrees.
    lons: array-like of shape (N,)
        1-D vector of longitude points expressed as degrees.

    Returns
    -------
    TFP: array-like of shape (..., M, N)
        Thermal front parameter expressed as degC/(100km)^2.

    Examples
    --------
    >>> np.random.seed(120)
    >>> field = np.random.uniform(low=0, high=40, size=(5, 5))  # random temperatures between 0 and 40 degrees Celsius
    >>> field
    array([[27.11822193, 20.51835209, 24.94822847, 19.0856986 , 18.41039256],
           [38.03459464, 39.38302396, 34.17690184, 23.6436138 ,  8.12785491],
           [10.4944062 ,  2.65660985, 25.87740026, 28.74931783, 14.04197021],
           [38.10173911, 23.81909712, 39.78024821, 21.7469418 ,  2.86850514],
           [ 5.62746737,  8.29113292, 20.22109633, 21.4157172 , 21.25820336]])
    >>> lons = np.arange(120, 201, 20)
    >>> lons
    array([120, 140, 160, 180, 200])
    >>> lats = np.arange(40, 61, 5)
    >>> lats
    array([40, 45, 50, 55, 60])
    >>> TFP = thermal_front_parameter(field, lats, lons)
    >>> TFP
    array([[-0.51556961, -0.57097894,  0.01174865, -0.06425258, -0.14175305],
           [ 0.00375789, -0.48249109,  0.15134022, -0.0408684 , -0.17021388],
           [-0.1660594 ,  0.20510328, -0.19987511, -0.06345793,  0.2326307 ],
           [-1.03967341, -0.40324022, -0.61269204,  0.11165267,  0.53391446],
           [-0.02207704,  0.01454242, -0.00466246, -0.00478738, -0.0053461 ]])
    """

    # convert lats and lons to cartesian coordinates so spatial gradients can be calculated
    Lons, Lats = np.meshgrid(lons, lats)
    x, y = data_utils.haversine(Lons, Lats)

    # gradient vector of the thermodynamic field
    dFdx = np.diff(field, axis=-1, append=0) / np.diff(x, axis=-1, append=0)
    dFdy = np.diff(field, axis=-2, append=0) / np.diff(y, axis=-2, append=0)
    dF = np.array([dFdx, dFdy])

    dFmag = np.sqrt(np.sum(np.square(dF), axis=0))  # magnitude of the gradient vector of the thermodynamic field
    dF_unit_vector = dF / dFmag  # unit vector in the direction of the gradient vector

    # gradient vector of the magnitude of the gradient vector of the thermodynamic field
    ddFmagdx = np.diff(dFmag, axis=-1, append=0) / np.diff(x, axis=-1, append=0)
    ddFmagdy = np.diff(dFmag, axis=-2, append=0) / np.diff(y, axis=-2, append=0)
    ddFmag = np.array([ddFmagdx, ddFmagdy])

    # calculate thermal front parameter and change units from degC/(km^2) to degC/(100km)^2
    TFP = np.sum(- ddFmag * dF_unit_vector, axis=0) * 1e4

    return TFP


def thermal_front_locator(field, lats, lons):
    """
    Calculates the thermal front locator (TFL; Huber-Pock & Kress 1981) using a provided thermodynamic variable.

    Parameter
    ---------
    field: array-like of shape (..., M, N)
        2-D field of the thermodynamic variable that will be used to calculate the TFL. The last two axes of length M and
            N must be the latitude and longitude axes, respectively.
    lats: array-like of shape (M,)
        1-D vector of latitude points expressed as degrees.
    lons: array-like of shape (N,)
        1-D vector of longitude points expressed as degrees.

    Returns
    -------
    TFL: array-like of shape (..., M, N)
        Thermal front locator expressed as degC/(100km)^4.

    Examples
    --------
    >>> np.random.seed(120)
    >>> field = np.random.uniform(low=0, high=40, size=(5, 5))  # random temperatures between 0 and 40 degrees Celsius
    >>> field
    array([[27.11822193, 20.51835209, 24.94822847, 19.0856986 , 18.41039256],
           [38.03459464, 39.38302396, 34.17690184, 23.6436138 ,  8.12785491],
           [10.4944062 ,  2.65660985, 25.87740026, 28.74931783, 14.04197021],
           [38.10173911, 23.81909712, 39.78024821, 21.7469418 ,  2.86850514],
           [ 5.62746737,  8.29113292, 20.22109633, 21.4157172 , 21.25820336]])
    >>> lons = np.arange(120, 201, 20)
    >>> lons
    array([120, 140, 160, 180, 200])
    >>> lats = np.arange(40, 61, 5)
    >>> lats
    array([40, 45, 50, 55, 60])
    >>> TFL = thermal_front_locator(field, lats, lons)
    >>> TFL
    array([[ 9.26191422e-02,  1.76187373e-02,  2.53556619e-02,
             4.34760673e-03,  5.07964100e-03],
           [ 3.02196913e-02, -1.24731530e-01,  6.27675627e-02,
             8.60028595e-04,  7.23569573e-02],
           [-1.58047350e-01, -1.10518190e-01, -7.37258066e-02,
            -3.46076518e-02, -5.41030478e-02],
           [-1.85557854e-01, -7.51105984e-02, -1.15051591e-01,
            -2.00581092e-02, -9.69546769e-02],
           [ 1.45060450e-03, -9.34552106e-04, -6.97467095e-05,
            -7.09751637e-05, -8.47798408e-05]])
    """

    # convert lats and lons to cartesian coordinates so spatial gradients can be calculated
    Lons, Lats = np.meshgrid(lons, lats)
    x, y = data_utils.haversine(Lons, Lats)

    # gradient vector of the thermodynamic field
    dFdx = np.diff(field, axis=-1, append=0) / np.diff(x, axis=-1, append=0)
    dFdy = np.diff(field, axis=-2, append=0) / np.diff(y, axis=-2, append=0)
    dF = np.array([dFdx, dFdy])

    dFmag = np.sqrt(np.sum(np.square(dF), axis=0))  # magnitude of the gradient vector of the thermodynamic field
    dF_unit_vector = dF / dFmag  # unit vector in the direction of the gradient vector

    # thermal front parameter expressed as degC/(100km)^2
    TFP = thermal_front_parameter(field, lats, lons)

    # gradient vector of the thermal front parameter
    dTFPx = np.diff(TFP, axis=-1, append=0) / np.diff(x, axis=-1, append=0)
    dTFPy = np.diff(TFP, axis=-2, append=0) / np.diff(y, axis=-2, append=0)
    dTFP = np.array([dTFPx, dTFPy]) * 100  # units: degC/(100km)^3

    # calculate thermal front parameter expressed as degC/(100km)^4
    TFL = np.sum(dTFP * dF_unit_vector, axis=0)

    return TFL


def minimum_maximum_locator(field, lats, lons):
    """
    Calculates the Minimum-Maximum locator (MML; Clarke and Renard 1966).

    Parameter
    ---------
    field: array-like of shape (..., M, N)
        2-D field of the thermodynamic variable that will be used to calculate the MML. The last two axes of length M and
            N must be the latitude and longitude axes, respectively.
    lats: array-like of shape (M,)
        1-D vector of latitude points expressed as degrees.
    lons: array-like of shape (N,)
        1-D vector of longitude points expressed as degrees.

    Returns
    -------
    MML: array-like of shape (..., M, N)
        Minimum-Maximum locator expressed as degC/(100km)^4.

    Examples
    --------
    >>> np.random.seed(120)
    >>> field = np.random.uniform(low=0, high=40, size=(5, 5))  # random temperatures between 0 and 40 degrees Celsius
    >>> field
    array([[27.11822193, 20.51835209, 24.94822847, 19.0856986 , 18.41039256],
           [38.03459464, 39.38302396, 34.17690184, 23.6436138 ,  8.12785491],
           [10.4944062 ,  2.65660985, 25.87740026, 28.74931783, 14.04197021],
           [38.10173911, 23.81909712, 39.78024821, 21.7469418 ,  2.86850514],
           [ 5.62746737,  8.29113292, 20.22109633, 21.4157172 , 21.25820336]])
    >>> lons = np.arange(120, 201, 20)
    >>> lons
    array([120, 140, 160, 180, 200])
    >>> lats = np.arange(40, 61, 5)
    >>> lats
    array([40, 45, 50, 55, 60])
    >>> MML = minimum_maximum_locator(field, lats, lons)
    >>> MML
    array([[ 1.97110331,  1.86558045,  1.68252513,  0.63611081,  1.82137088],
           [ 3.87441325, -6.46895958,  1.55136154,  0.13647153,  1.06292906],
           [-4.97573923, -3.95079091, -2.47669703, -1.44947791, -2.00722563],
           [-5.87638048, -2.87788414, -3.62502599, -0.64223281, -3.30506401],
           [ 0.12176053, -0.57844794, -0.30743987, -0.29444804, -0.33711797]])
    """

    # convert lats and lons to cartesian coordinates so spatial gradients can be calculated
    Lons, Lats = np.meshgrid(lons, lats)
    x, y = data_utils.haversine(Lons, Lats)

    # gradient vector of the thermodynamic field
    dFdx = np.diff(field, axis=-1, append=0) / np.diff(x, axis=-1, append=0)
    dFdy = np.diff(field, axis=-2, append=0) / np.diff(y, axis=-2, append=0)
    dF = np.array([dFdx, dFdy])

    # thermal front parameter expressed as degC/(100km)^2
    TFP = thermal_front_parameter(field, lats, lons)

    # gradient vector of the thermal front parameter
    dTFPx = np.diff(TFP, axis=-1, append=0) / np.diff(x, axis=-1, append=0)
    dTFPy = np.diff(TFP, axis=-2, append=0) / np.diff(y, axis=-2, append=0)
    dTFP = np.array([dTFPx, dTFPy]) * 100  # units: degC/(100km)^3

    dTFPmag = np.sqrt(np.sum(np.square(dTFP), axis=0))  # magnitude of the gradient vector of the TFP
    dTFP_unit_vector = dTFP / dTFPmag  # unit vector in the direction of the gradient vector

    # minimum-maximum locator expressed as degC/(100km)^4
    MML = np.sum(dF * dTFP_unit_vector, axis=0) * 100

    return MML