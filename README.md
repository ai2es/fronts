## Environment Setup
You can setup the environment **two** different ways:
1. Using the requirements.txt file
2. Manually create the environment with pip and conda

### 1. Setup with requirements.txt file
```commandline
conda create -n <env> -f requirements.txt
```

### 2. Setup the environment with conda and pip
1. Create the initial environment:

```commandline
conda create -n <env> python=3.10
```

2. Install mamba (a package manager that is much faster than conda):
```commandline
conda install mamba -c conda-forge
```

3. Install TensorFlow 
* If you need GPU compatibility:
```commandline
mamba install cudatoolkit=11.2.2 cudnn=8.1.0.77 -c conda-forge
pip install tensorflow==2.10

# verify that your GPU(s) can be found
python3 -c "import tensorflow as tf; print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))"
```

* All other users:
```commandline
pip install tensorflow
```

4. Install the rest of the required packages
```commandline
mamba install cartopy cfgrib folium matplotlib netcdf4 pandas scikit-learn scipy shapely streamlit xarray -c conda-forge
pip install geojsoncontour wget 
```