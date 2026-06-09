import argparse
import dataclasses
import logging
import sys

import icechunk as ic
import icechunk.xarray
import pandas as pd
import tqdm
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


@dataclasses.dataclass
class StoreContents:
    """Snapshot of what is currently present in an icechunk store.

    Attributes:
        variables: Data variable names found in the store.
        times: DatetimeIndex of time steps present.
        levels: Pressure levels stored.
        coordinates: Spatial bounding box of stored data.
    """

    variables: list[str]
    times: pd.DatetimeIndex
    levels: list[int]
    coordinates: utils.BoundingBox


@dataclasses.dataclass
class WriteStrategy:
    """Decision record produced by determine_write_strategy.

    Exactly one of skip_reason, error_reason, or an action field will be populated:
    - skip_reason non-empty: log and return immediately.
    - error_reason non-empty: raise ValueError with the reason.
    - Otherwise: proceed using missing_variables and/or missing_times.

    Attributes:
        missing_variables: Variables absent from the store that need to be added.
        missing_times: Time steps present in the request but absent from the store.
        skip_reason: Non-empty when no write is needed; human-readable explanation.
        error_reason: Non-empty when write is impossible; raised as ValueError.
    """

    missing_variables: list[str]
    missing_times: pd.DatetimeIndex
    skip_reason: str
    error_reason: str
    merge_required: bool

    def execute(
        self,
        era5_config: config.ERA5DataLoaderConfig,
        icechunk_config: config.IcechunkStorageConfig,
    ) -> None:
        """Generate and write the ERA5 data described by this strategy.

        Args:
            era5_config: Base ERA5 configuration; time range or variables will be
                narrowed to only what is missing before generating data.
            icechunk_config: Configuration for the target icechunk store.
        """
        if self.missing_times.size:
            if self.merge_required:
                logger.info("Generating ERA5 data for missing time steps...")
                missing_ds = generate_era5_data(era5_config).sel(time=self.missing_times)
                logger.info("Appending missing time steps to icechunk store...")
                write_or_append_icechunk_store(icechunk_config, missing_ds)
            else:
                data_config = dataclasses.replace(
                    era5_config,
                    time_start=self.missing_times[0].to_pydatetime(),
                    time_end=self.missing_times[-1].to_pydatetime(),
                )
                logger.info("Generating ERA5 data subset...")
                era5_subset = generate_era5_data(data_config)
                logger.info("Writing ERA5 subset to icechunk store...")
                write_or_append_icechunk_store(icechunk_config, era5_subset)
        else:
            data_config = dataclasses.replace(era5_config, variables=self.missing_variables)
            logger.info("Generating ERA5 data subset...")
            era5_subset = generate_era5_data(data_config)
            logger.info("Writing ERA5 subset to icechunk store...")
            write_new_variables_to_icechunk_store(icechunk_config, era5_subset)


def generate_era5_data(config: config.ERA5DataLoaderConfig) -> xr.Dataset:
    """Generate a subset of ERA5 data for model training, validation, and testing.

    Args:
        config: Configuration object containing all necessary parameters for data
            generation.

    Returns:
        An xarray Dataset containing the subset of ERA5 data specified by the config.
    """
    open_kwargs: dict = {"chunks": None}
    if config.storage_options:
        open_kwargs["storage_options"] = config.storage_options
    ds = xr.open_zarr(config.era5_uri, **open_kwargs)

    ds_variable_subset = ds[config.variables]
    date_range = pd.date_range(config.time_start, config.time_end, freq=config.time_resolution)

    # Attach periodic index to longitude for > 360 to avoid wrap-crossing slice issues
    ds_variable_subset = utils.attach_periodic_lon_index(ds_variable_subset)
    ds_full_subset = ds_variable_subset.sel(
        time=date_range,
        latitude=slice(config.coordinates.lat_max, config.coordinates.lat_min),
        longitude=slice(config.coordinates.lon_min, config.coordinates.lon_max),
        level=config.pressure_levels,
    )
    return ds_full_subset.chunk(config.chunks)


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

    if ic.Repository.exists(storage):
        append = "time"
        logger.info(f"Opening existing icechunk repository at {storage_config.store_path}")
    else:
        append = None
        logger.info(f"Creating new icechunk repository at {storage_config.store_path}")
    repo = ic.Repository.open_or_create(storage)
    return repo, append


def inspect_store(storage_config: config.IcechunkStorageConfig) -> StoreContents | None:
    """Return a snapshot of the variables, times, levels, and spatial extent in the store.

    Args:
        storage_config: Configuration for the icechunk store.

    Returns:
        StoreContents if the store exists, None otherwise.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)
    if not ic.Repository.exists(storage):
        return None
    ds = utils.open_readonly_icechunk_store(storage_config.store_path, storage_config.branch_name)
    ds = utils.unwrap_longitude(ds)
    return StoreContents(
        variables=[str(k) for k in ds.data_vars],
        times=pd.DatetimeIndex(ds.time.values),
        levels=list(map(int, ds.level.values)),
        coordinates=utils.BoundingBox(
            lat_min=float(ds.latitude.min()),
            lat_max=float(ds.latitude.max()),
            lon_min=float(ds.longitude.min()),
            lon_max=float(ds.longitude.max()),
        ),
    )


def determine_write_strategy(
    era5_config: config.ERA5DataLoaderConfig,
    store_contents: StoreContents | None,
) -> WriteStrategy:
    """Determine what needs to be written or appended to the icechunk store.

    Args:
        era5_config: Configuration describing what data is requested.
        store_contents: Current state of the store, or None if the store does not exist.

    Returns:
        WriteStrategy describing what action to take.
    """
    requested_times = pd.date_range(era5_config.time_start, era5_config.time_end, freq=era5_config.time_resolution)
    missing_variables: list[str] = []
    missing_times: pd.DatetimeIndex = pd.DatetimeIndex([])
    skip_reason = ""
    error_reason = ""
    merge_required = False

    if store_contents is None:
        missing_variables = list(era5_config.variables)
        missing_times = requested_times
    elif list(era5_config.pressure_levels) != store_contents.levels or (
        era5_config.coordinates != store_contents.coordinates
    ):
        error_reason = (
            "Non-time dimension mismatch between config and store. "
            "Delete the store and rerun to regenerate with the new configuration."
        )
    else:
        missing_variables = [v for v in era5_config.variables if v not in store_contents.variables]
        missing_times = requested_times[~requested_times.isin(store_contents.times)]

        if len(missing_variables) > 0 and len(missing_times) > 0:
            error_reason = (
                "Store has both missing variables and missing time steps. "
                "Add the missing time steps first, then rerun to add new variables."
            )
        elif len(missing_variables) == 0 and len(missing_times) == 0:
            skip_reason = "All requested variables and time steps are already present in the store."
        elif len(missing_times) > 0:
            if len(store_contents.times) == 0:
                error_reason = "Store exists but contains no time steps. Delete the store and rerun."
            elif missing_times[0] <= store_contents.times[-1]:
                merge_required = True

    return WriteStrategy(
        missing_variables=missing_variables,
        missing_times=missing_times,
        skip_reason=skip_reason,
        error_reason=error_reason,
        merge_required=merge_required,
    )


def write_new_variables_to_icechunk_store(
    storage_config: config.IcechunkStorageConfig,
    ds: xr.Dataset | xr.DataArray,
) -> None:
    """Write new variables into an existing icechunk store.

    Opens the existing store to validate that the incoming dataset's dimensions
    are identical, then writes a minimal Dataset containing only the new
    variable arrays (no coordinate arrays) so existing dimension coordinates
    are never re-written or extended.

    Args:
        storage_config: Configuration for the icechunk store.
        ds: Dataset containing only the new variables to add.

    Raises:
        ValueError: If the dimensions or shape of ``ds`` do not exactly match
            those already present in the store.
    """
    storage = ic.local_filesystem_storage(storage_config.store_path)
    repo = ic.Repository.open(storage)

    existing_ds = utils.open_readonly_icechunk_store(
        storage_config.store_path,
        storage_config.branch_name,
        group=storage_config.group_name,
        zarr_format=storage_config.zarr_format,
    )
    if dict(ds.sizes) != dict(existing_ds.sizes):
        raise ValueError(f"Dimension mismatch: incoming {dict(ds.sizes)} != store {dict(existing_ds.sizes)}")

    logger.info(
        f"Adding new variables {list(ds.data_vars)} to icechunk store at "
        f"{storage_config.store_path} with dataset of shape {ds.sizes}"
    )

    new_ds = xr.Dataset(
        {
            name: xr.DataArray(ds[name].drop_encoding().data, dims=list(ds[name].dims), attrs=ds[name].attrs)
            for name in ds.data_vars
        }
    )
    session = repo.writable_session(storage_config.branch_name)
    icechunk.xarray.to_icechunk(new_ds, session, mode="a", safe_chunks=False)
    log = f"{storage_config.commit_message} - added variables: {list(ds.data_vars)}"
    session.commit(log)
    logger.info(log)


def write_or_append_icechunk_store(
    storage_config: config.IcechunkStorageConfig, ds: xr.Dataset | xr.DataArray, dry_run: bool = False
) -> None:
    """Write or append an xarray Dataset to an icechunk store.

    If the store already exists, the new data will be appended along the time dimension.

    Args:
        storage_config: Configuration object containing parameters for icechunk storage.
        ds: The xarray Dataset to write or append to the store.
        dry_run: If True, perform a dry run without actually writing to the store.
    """
    # Drop encoding due to zarr v3 compatibility issues; see
    # https://github.com/pydata/xarray/issues/10032
    ds = ds.drop_encoding()
    global_attrs = ds.attrs

    repo, append = get_local_icechunk_repository(storage_config)

    logger.info(
        f"{'Appending to' if append else 'Writing new'} icechunk store at "
        f"{storage_config.store_path} with variables {list(ds.data_vars)} and shape {ds.sizes}"
    )

    time_chunk_size = 1460
    times = ds.time.values
    chunks = range(0, len(times), time_chunk_size)
    with tqdm.tqdm(chunks, unit="year", desc="Writing icechunk") as pbar:
        for i in pbar:
            time_slice = times[i : i + time_chunk_size]
            ds_slice = ds.sel(time=time_slice)
            ds_slice.attrs = global_attrs

            pbar.set_postfix(year=str(time_slice[0])[:4], steps=f"{i}-{i + len(time_slice)}")
            session = repo.writable_session(storage_config.branch_name)
            icechunk.xarray.to_icechunk(ds_slice, session, append_dim=append, safe_chunks=False)
            if dry_run:
                logger.info(
                    f"Dry run: would commit {len(time_slice)} time steps from {time_slice[0]} to {time_slice[-1]}"
                )
            else:
                session.commit(f"{storage_config.commit_message}, time steps {i} to {i + len(time_slice)}")
            append = "time"  # always append after first write


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

    bounding_box_type_hook = {utils.BoundingBox: lambda d: utils.BoundingBox(*d)}

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

    store_contents = inspect_store(icechunk_config)
    strategy = determine_write_strategy(era5_config, store_contents)

    if strategy.error_reason:
        raise ValueError(strategy.error_reason)
    if strategy.skip_reason:
        logger.info(strategy.skip_reason)
        return

    logger.info(f"Variables to add:      {strategy.missing_variables}")
    logger.info(f"Time steps to add:     {len(strategy.missing_times)}")
    strategy.execute(era5_config, icechunk_config)
    logger.info("Data generation and storage complete.")


if __name__ == "__main__":
    main()
