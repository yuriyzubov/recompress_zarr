import zarr 
import click
import pydantic_zarr as pz
from numcodecs import Zstd
import copy_data as cd
import time



__version__ = "0.1.0"

def copy_n5_tree(n5_root, z_store):
    spec_n5 = pz.GroupSpec.from_zarr(n5_root)
    spec_n5_dict = spec_n5.dict()
    #normalize_groupspec(spec_n5_dict, comp)
    spec_n5 = pz.GroupSpec(**spec_n5_dict)
    return spec_n5.to_zarr(z_store, path= '')


def import_datasets(n5src, zarrdest, max_dask_chunk_num, compressor, scheduler, num_cores):

    old_store = zarr.NestedDirectoryStore(n5src)
    old_root = zarr.open_group(old_store, mode = 'r')
    zarr_arrays = (old_root.arrays(recurse=True))

    new_store = zarr.NestedDirectoryStore(zarrdest)
    new_root = copy_n5_tree(old_root, new_store)

    start_time = time.time()

    copy_arrays_data = cd.cluster_compute(scheduler, num_cores)(cd.copy_arrays)
    copy_arrays_data(zarr_arrays, zarrdest, max_dask_chunk_num, compressor)

    total_time = time.time() - start_time
    print(f"Total time is: {total_time} ms.")
    


@click.command()
@click.argument('src', type=click.STRING)
@click.argument('dest', type=click.STRING)
@click.option('--scheduler', default='local' ,type=click.STRING)
@click.option('--clevel', default = 6, type=click.INT)
@click.option('--darr_chunk_num', default = 50000, type=click.INT)
@click.option('--num_cores', '-c', default = 20, type=click.INT)
def cli(src, dest, scheduler, clevel, darr_chunk_num, num_cores):

    compressor = Zstd(level=clevel)
    import_datasets(src, dest, darr_chunk_num, compressor, scheduler, num_cores)

    
if __name__ == '__main__':
    cli()
