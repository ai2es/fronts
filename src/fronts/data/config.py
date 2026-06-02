import dataclasses
import datetime

from fronts import utils


@dataclasses.dataclass
class IcechunkStorageConfig:
    """Configuration for storing generated data in an Icechunk repository.

    Attributes:
        store_path: Path to the Icechunk store.
        branch_name: Name of the branch to write to.
        commit_message: Message for the commit.
        zarr_format: Zarr format version to use when writing the store. Default is 3.
        group_name: Optional group name within the zarr store. Default is None.
        virtual_chunk_local_path: If the store contains virtual chunks referencing local
            netcdf files, set this to the directory those files live in (e.g.
            ``/ourdisk/hpc/ai2es/tman/data/netcdf/``). The ``file://`` URL prefix is
            derived automatically. Leave None for stores with no virtual chunks.
    """

    store_path: str
    branch_name: str
    commit_message: str = "add data"
    zarr_format: int = 3
    group_name: str | None = None
    virtual_chunk_local_path: str | None = None


@dataclasses.dataclass
class ERA5DataLoaderConfig:
    """Configuration for downloading remote ERA5 data for model training and evaluation.

    Attributes:
        era5_uri: URI to the ERA5 data in Zarr format.
        variables: List of variable names to load from the ERA5 dataset.
        pressure_levels: List of pressure levels to load for each variable.
        time_start: Start of the time range to load.
        time_end: End of the time range to load.
        time_resolution: Temporal resolution of the data (e.g., "6h" for
            6-hourly data).
        coordinates: Spatial bounding box to subset the data in order
            latitude min, latitude max, longitude min, longitude max.
        storage_options: Optional dictionary of storage options for xarray's open_zarr.
        chunks: Dictionary specifying chunk sizes after subsetting, e.g.,
            {"time": 100, "latitude": 64, "longitude": 64}.
    """

    era5_uri: str
    variables: list[str]
    pressure_levels: list[int]
    time_start: datetime.datetime
    time_end: datetime.datetime
    time_resolution: str
    coordinates: utils.BoundingBox
    storage_options: dict | None
    chunks: dict[str, int]


@dataclasses.dataclass
class DataConfig:
    """Configuration for loading and splitting ERA5 and fronts data.

    Attributes:
        era5_icechunk_config: Icechunk store config for ERA5 data.
        fronts_icechunk_config: Icechunk store config for fronts data.
        train_split: Fraction of time steps to use for training; remaining go to validation.
        batch_size: Number of timesteps per training batch.
        class_weights: Per-class loss weights. None means equal weighting.
        front_dilation: Number of binary dilation iterations applied to each non-background
            front class. 0 means no dilation.
    """

    era5_icechunk_config: IcechunkStorageConfig
    fronts_icechunk_config: IcechunkStorageConfig
    variables: list[str]
    train_split: float = 0.8
    batch_size: int = 4
    steps_per_epoch: int | None = None
    load_chunk_steps: int | None = None
    prefetch_chunks: int = 2
    class_weights: list[float] | None = None
    front_dilation: int = 0


@dataclasses.dataclass
class EvalConfig:
    """Configuration for running performance statistics evaluation.

    Attributes:
        model_path: Path to the saved .keras model checkpoint.
        outdir: Directory to write stats_aggregate_{mask}.nc and stats_spatial_{mask}.nc.
        front_types: Front type labels in class order (excluding background class 0).
        mask: Restrict statistics to "land" or "ocean" grid points. None means all points.
        front_dilation: Binary dilation iterations applied to truth labels. None uses
            the value from the paired DataConfig.
        coordinates: Spatial bounding box as [lat_min, lat_max, lon_min, lon_max].
            Defaults to CONUS extent.
        gpu_device: GPU index to use. None runs on CPU.
        time_start: Restrict evaluation to timesteps on or after this date. None means no lower bound.
        time_end: Restrict evaluation to timesteps before this date. None means no upper bound.
    """

    model_path: str
    outdir: str
    coordinates: utils.BoundingBox = dataclasses.field(
        default_factory=lambda: utils.BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
    )
    front_types: list[str] = dataclasses.field(default_factory=lambda: ["CF", "WF", "SF", "OF", "DL"])
    mask: str | None = None
    front_dilation: int | None = None
    gpu_device: int | None = None
    time_start: datetime.datetime | None = None
    time_end: datetime.datetime | None = None


@dataclasses.dataclass
class PredictConfig:
    """Configuration for generating a single-timestep case study prediction plot.

    Attributes:
        model_path: Path to the saved .keras model checkpoint.
        outdir: Directory to write the output PNG.
        init_time: Timestep as [year, month, day, hour].
        front_types: Front type labels in class order (excluding background class 0).
        coordinates: Spatial bounding box as [lat_min, lat_max, lon_min, lon_max].
            Defaults to full domain extent.
        prob_mask: Minimum probability to display; values below are masked.
        prob_interval: Contour interval for probability levels.
        filled_contours: Plot filled probability contours.
        open_contours: Plot open (line) probability contours.
        targets: Overlay ground truth fronts from the icechunk fronts store.
        gpu_device: GPU index to use. None runs on CPU.
    """

    model_path: str
    outdir: str
    init_time: datetime.datetime
    front_types: list[str] = dataclasses.field(default_factory=lambda: ["CF", "WF", "SF", "OF", "DL"])
    coordinates: utils.BoundingBox = dataclasses.field(
        default_factory=lambda: utils.BoundingBox(lat_min=0.25, lat_max=80.0, lon_min=130.0, lon_max=369.75)
    )
    prob_mask: float = 0.1
    prob_interval: float = 0.1
    filled_contours: bool = False
    open_contours: bool = False
    targets: bool = False
    gpu_device: int | None = None
