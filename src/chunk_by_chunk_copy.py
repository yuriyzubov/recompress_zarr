import json
import os
from typing import Literal, Tuple, Union
from dask.distributed import wait
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster


import dask.array as da
import zarr
#import cluster_wrapper as cw
import time
from numcodecs.abc import Codec
import numpy as np
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client
from numcodecs import Zstd
from toolz import partition_all





src_store = zarr.DirectoryStore('')
src = zarr.open(store=src_store, path='/mito_membrane_postprocessed', mode = 'r')

dest_store = zarr.NestedDirectoryStore('')
dest_root = zarr.open_group(store=dest_store, mode= 'a')

def save_chunk(
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...]):
    
    in_slices = tuple(out_slice for out_slice in out_slices)
    source_data = source[in_slices]
    # only store source_data if it is not all 0s
    if not (source_data == 0).all():
        dest[out_slices] = source_data
    return 1

def copy_arrays(z_src: zarr.Group,
                dest_root: zarr.Group,
                client: Client,
                num_workers: int,
                comp: Codec):
    
    
    # store original array in a new .zarr file as an arr_name
    client.cluster.scale(num_workers)
    if isinstance(z_src, zarr.core.Array):
        z_arrays = [z_src]
    else:
        z_arrays = [key_val_arr[1] for key_val_arr in (z_src.arrays())]
    for src_arr in z_arrays:
        

        start_time = time.time()
        dest_arr = dest_root.require_dataset(
            src_arr.name, 
            shape=src_arr.shape, 
            chunks=src_arr.chunks, 
            dtype=src_arr.dtype, 
            compressor=src_arr.compressor, 
            dimension_separator='/')#,
            # fill_value=0,
            # exact=True)

        out_slices = slices_from_chunks(normalize_chunks(dest_arr.chunks, shape=dest_arr.shape))
        # break the slices up into batches, to make things easier for the dask scheduler
        out_slices_partitioned = tuple(partition_all(100000, out_slices))
        for idx, part in enumerate(out_slices_partitioned):
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: save_chunk(src_arr, dest_arr, v), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')

# def add_multiscale_metadata(dest_root):
#     z_attrs['multiscales'][0]['name'] = dest_root.name
#     return z_attrs

if __name__ == '__main__':
    # store_multiscale = cw.cluster_compute("local")(create_multiscale)
    # store_multiscale(z_src,(64, 64, 64),  Zstd(level=6))
    num_cores = 1
    cluster = LSFCluster(
        cores=num_cores,
        processes=num_cores,
        memory=f"{15 * num_cores}GB",
        ncpus=num_cores,
        mem=15 * num_cores,
        walltime="48:00",
        local_directory = "/scratch/$USER/"
        )
    
    #cluster = LocalCluster()
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)

    copy_arrays(z_src=src, dest_root=dest_root, client=client, num_workers=500, comp=Zstd(level=6))