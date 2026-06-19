import dataclasses
import datetime

from fronts import utils


@dataclasses.dataclass
class SlurmConfig:
    """Configuration for the dask-jobqueue SLURMCluster used during derivation.

    Attributes:
        queue: SLURM partition name (e.g. "ai2es").
        cores: Total CPU cores per SLURM job.
        processes: Dask worker processes per job.
        memory: Memory string per job (e.g. "128GB").
        walltime: Wall time string (e.g. "12:00:00").
        stdout: Path template for worker stdout logs. Use ``%j`` for the SLURM
            job ID (e.g. ``/path/to/logs/generate_%j_out.txt``).
        stderr: Path template for worker stderr logs. Use ``%j`` for the SLURM
            job ID (e.g. ``/path/to/logs/generate_%j_err.txt``).
        n_jobs: Number of SLURM jobs to scale the cluster to.
    """

    queue: str
    cores: int
    processes: int
    memory: str
    walltime: str
    stdout: str
    stderr: str
    n_jobs: int


@dataclasses.dataclass
class EvalConfig:
    """Configuration for running performance statistics evaluation.

    Attributes:
        model_path: Path to the saved .keras model checkpoint.
        outdir: Directory to write stats_aggregate_{mask}.nc and stats_spatial_{mask}.nc.
        coordinates: Spatial bounding box as [lat_min, lat_max, lon_min, lon_max].
            Defaults to full USAD extent.
        front_types: Front type labels in class order (excluding background class 0).
        mask: Restrict statistics to "land" or "ocean" grid points. None means all points.
        front_dilation: Binary dilation iterations applied to truth labels. None uses
            the value from the paired DatasetConfig.
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
