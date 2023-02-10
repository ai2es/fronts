import os
import argparse
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help="Model directory")
    parser.add_argument('--netcdf_dir', type=str, help="netCDF directory")

    args = parser.parse_args()

    if args.model_dir is not None:
        files_to_delete = glob('%s/model_*/predictions/*.nc' % args.model_dir)
        if len(files_to_delete) > 0:
            print(f"Deleting %d prediction files in {args.model_dir}" % len(files_to_delete))
            for file in files_to_delete:
                os.remove(file)
        else:
            print(f"No prediction files found in {args.model_dir}")

    if args.netcdf_dir is not None:
        files_to_delete = glob('%s/*.nc' % args.netcdf_dir)
        if len(files_to_delete) > 0:
            print(f"Deleting %d netCDF files in {args.netcdf_dir}" % len(files_to_delete))
            for file in files_to_delete:
                os.remove(file)
        else:
            print(f"No netCDF files found in {args.netcdf_dir}")

