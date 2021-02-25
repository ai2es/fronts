import os
import argparse

def run_create_NC_files(start_year, end_year, start_month, end_month, start_day, end_day, netcdf_outdir, image_outdir):
    for a in range(start_year,end_year+1):
     year = a
     for b in range(start_month,end_month+1):
         month = b
         for c in range(start_day,end_day+1):
             day = c
             os.system("python create_NC_files.py --year %d --month %d --day %d --netcdf_outdir %s --image_outdir %s" % (year,
                month, day, netcdf_outdir, image_outdir))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create labeled data for the specified day")
    parser.add_argument('--start_year', type=int, required=True, help="starting year for the data to be read in")
    parser.add_argument('--end_year', type=int, required=True, help="end year for the data to be read in")
    parser.add_argument('--start_month', type=int, required=True, help="starting month for the data to be read in")
    parser.add_argument('--end_month', type=int, required=True, help="ending month for the data to be read in")
    parser.add_argument('--start_day', type=int, required=True, help="starting day for the data to be read in")
    parser.add_argument('--end_day', type=int, required=True, help="ending day for the data to be read in")
    parser.add_argument('--image_outdir', type=str, required=True, help="output directory for image files")
    parser.add_argument('--netcdf_outdir', type=str, required=True, help="output directory for netcdf files")
    args = parser.parse_args()
    # read the polygons for the specified day
    run_create_NC_files(args.start_year, args.end_year, args.start_month, args.end_month, args.start_day,
                    args.end_day, args.netcdf_outdir, args.image_outdir)
