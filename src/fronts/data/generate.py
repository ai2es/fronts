import argparse
import dataclasses
import logging
import sys

import icechunk as ic
import icechunk.xarray
import pandas as pd
import xarray as xr

from fronts import utils
from fronts.data import config

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def generate_era5_data(config: config.ERA5DataLoaderConfig) -> xr.Dataset | xr.DataArray:
    """Generate a subset of ERA5 data for model training, validation, and testing.

    Args:
        config: Configuration object containing all necessary parameters for data
            generation.

    Returns:
        An xarray Dataset containing the subset of ERA5 data specified by the config.
    """
    # Use xarray's open_zarr with chunks set to None for opening
    open_kwargs = {"chunks": None}
    if config.storage_options:
        open_kwargs["storage_options"] = config.storage_options
    ds = xr.open_zarr(config.era5_uri, **open_kwargs)

    # Subset variables, time range, and spatial bounding box according to config
    ds_variable_subset = ds[config.variables]
    date_range = pd.date_range(config.time_start, config.time_end, freq=config.time_resolution)
    # Attach periodic index to longitude for > 360 to avoid wrap-crossing slice issues,
    # then rechunk after subsetting
    ds_variable_subset = utils.attach_periodic_lon_index(ds_variable_subset)
    ds_full_subset = ds_variable_subset.sel(
        time=date_range,
        latitude=slice(config.coordinates.lat_max, config.coordinates.lat_min),
        longitude=slice(config.coordinates.lon_min, config.coordinates.lon_max),
        level=config.pressure_levels,
    )
    ds_full_subset = ds_full_subset.chunk(config.chunks)

    return ds_full_subset


def get_local_icechunk_repository(
    storage_config: config.IcechunkStorageConfig,
) -> tuple[ic.Repository, str | None]:
    """Open or create an icechunk repository for storing processed data.

    Args:
        storage_config: Configuration object containing parameters for icechunk storage.

    Returns:
        An icechunk Repository object ready for use.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)

    # If the repository already exists, we will append to it; otherwise, we will create
    # a new one
    if ic.Repository.exists(storage):
        append = "time"
        logger.info(f"Opening existing icechunk repository at {storage_config.store_path}")
    else:
        append = None
        logger.info(f"Creating new icechunk repository at {storage_config.store_path}")
    repo = ic.Repository.open_or_create(storage)
    return repo, append


def get_existing_variables(storage_config: config.IcechunkStorageConfig) -> list[str]:
    """Return the list of data variable names already present in an icechunk store.

    Args:
        storage_config: Configuration for the icechunk store.

    Returns:
        Variable names in the store, or an empty list if the store does not exist.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)
    if not ic.Repository.exists(storage):
        return []
    return list(utils.open_readonly_icechunk_store(storage_config.store_path, storage_config.branch_name).data_vars)


def write_new_variables_to_icechunk_store(
    storage_config: config.IcechunkStorageConfig,
    ds: xr.Dataset | xr.DataArray,
) -> None:
    """Write new variables into an existing icechunk store.

    Uses zarr append mode so existing variables and dimension coordinates are
    left untouched.  All time steps in ``ds`` must already be present in the
    store's time coordinate.

    Args:
        storage_config: Configuration for the icechunk store.
        ds: Dataset containing only the new variables to add.
    """
    ds = ds.drop_encoding()
    storage = ic.local_filesystem_storage(storage_config.store_path)
    repo = ic.Repository.open(storage)

    logger.info(
        f"Adding new variables {list(ds.data_vars)} to icechunk store at "
        f"{storage_config.store_path} with dataset of shape {ds.sizes}"
    )

    session = repo.writable_session(storage_config.branch_name)
    icechunk.xarray.to_icechunk(ds, session, mode="a", safe_chunks=False)
    session.commit(f"{storage_config.commit_message} - added variables: {list(ds.data_vars)}")
    logger.info(f"Committed new variables: {list(ds.data_vars)}")


def write_or_append_icechunk_store(storage_config: config.IcechunkStorageConfig, ds: xr.Dataset | xr.DataArray) -> None:
    """Write or append an xarray Dataset to an icechunk store.

    If the store already exists, the new data will be appended along the time dimension.

    Args:
        storage_config: Configuration object containing parameters for icechunk storage.
        ds: The xarray Dataset to write or append to the store.
    """
    # Drop encoding due to zarr v3 compatibility issues; see
    # https://github.com/pydata/xarray/issues/10032
    ds = ds.drop_encoding()

    # Appends if existing, if not creates new store with the given data
    repo, append = get_local_icechunk_repository(storage_config)

    logger.info(
        f"{'Appending to' if append else 'Writing new'} icechunk store at "
        f"{storage_config.store_path} with dataset of shape {ds.sizes}"
    )

    # Write in yearly chunks to bound memory usage
    time_chunk_size = 1460  # ~1 year at 6h resolution
    times = ds.time.values

    for i in range(0, len(times), time_chunk_size):
        time_slice = times[i : i + time_chunk_size]
        ds_slice = ds.sel(time=time_slice)

        session = repo.writable_session(storage_config.branch_name)
        ds_slice = ds_slice.drop_encoding()

        icechunk.xarray.to_icechunk(ds_slice, session, append_dim=append, safe_chunks=False)
        session.commit(f"{storage_config.commit_message}, time steps {i} to {i + len(time_slice)}")
        append = "time"  # always append after first write

        logger.info(f"Committed time steps {i} to {i + len(time_slice)}")


def create_dask_client(
    scheduler_options: dict | None = None,
):
    """Create a Dask client for parallel processing.

    This function sets up a Dask cluster using the SLURM job scheduler and returns
    a client connected to that cluster. Adjust the cluster parameters as needed for
    your specific computing environment.

    Returns:
        A Dask Client object connected to the created cluster.
    """
    import dask_jobqueue  # type: ignore
    from dask.distributed import Client

    if scheduler_options is None:
        scheduler_options = {"dashboard_address": ":13921", "interface": "ib0"}

    slurm_cluster = dask_jobqueue.SLURMCluster(
        queue="ai2es",
        cores=64,
        processes=16,
        memory="128GB",
        walltime="12:00:00",
        shebang="#!/usr/bin/bash",
        log_directory="/ourdisk/hpc/ai2es/tman/logs/",
        scheduler_options=scheduler_options,
    )
    slurm_cluster.scale(jobs=1)
    client = Client(slurm_cluster)
    return client


def main():
    """Entry point for generating ERA5 icechunk data from config."""
    parser = argparse.ArgumentParser(description="Generate ERA5 data and store in icechunk")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML data generation config")
    args = parser.parse_args()

    # client = create_dask_client()
    # logger.info(f"Dask client created with dashboard at {client.dashboard_link}")

    # Add type hook for utils.BoundingBox to properly parse it from YAML config
    bounding_box_type_hook = {utils.BoundingBox: lambda d: utils.BoundingBox(*d)}

    # Open configs from YAML file
    era5_config = utils.open_config_yaml_as_dataclass(
        args.config,
        config.ERA5DataLoaderConfig,
        config_key="era5_config",
        type_hooks=bounding_box_type_hook,
    )
    logger.info(f"ERA5 data config loaded: {era5_config}")

    icechunk_config = utils.open_config_yaml_as_dataclass(
        args.config, config.IcechunkStorageConfig, config_key="icechunk_storage_config"
    )
    logger.info(f"Icechunk storage config loaded: {icechunk_config}")

    existing_variables = get_existing_variables(icechunk_config)
    missing_variables = [v for v in era5_config.variables if v not in existing_variables]

    if existing_variables and not missing_variables:
        logger.info("All requested variables already present in icechunk store. Nothing to download.")
        return
    if missing_variables != era5_config.variables:
        logger.info(f"Store contains: {existing_variables}. Downloading only missing: {missing_variables}")
        era5_config = dataclasses.replace(era5_config, variables=missing_variables)

    logger.info("Generating ERA5 data subset...")
    era5_subset = generate_era5_data(era5_config)
    logger.info("Writing ERA5 subset to icechunk store...")
    write_func = write_new_variables_to_icechunk_store if existing_variables else write_or_append_icechunk_store
    write_func(icechunk_config, era5_subset)
    logger.info("Data generation and storage complete.")


if __name__ == "__main__":
    main()
