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
#from numcodecs.gzip import Gzip
from toolz import partition_all
import sys

import click


def save_chunk_zarr(
        source: zarr.Array, 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        invert: bool):
    
    in_slices = tuple(out_slice for out_slice in out_slices)
    source_data = source[in_slices]
    # only store source_data if it is not all 0s
    if not (source_data == 0).all():
        if invert == True:
            dest[out_slices] = np.invert(source_data)
        else:
            dest[out_slices] = source_data
    return 1

def copy_arrays(z_src: zarr.Group | zarr.Array,
                dest_root: zarr.Group,
                client: Client,
                num_workers: int,
                comp: Codec,
                invert: bool,
                out_dtype: str):
    
    
    # store original array in a new .zarr file as an arr_name
    client.cluster.scale(num_workers)
    if isinstance(z_src, zarr.core.Array):
        z_arrays = [z_src]
    else:
        z_arrays = [key_val_arr[1] for key_val_arr in (z_src.arrays())]
        
    for src_arr in z_arrays:
        
        if out_dtype=='':
            out_dtype = src_arr.dtype
            
        if comp==None:
            comp = src_arr.compressor

        start_time = time.time()
        dest_arr = dest_root.require_dataset(
            src_arr.name, 
            shape=src_arr.shape, 
            chunks=src_arr.chunks, 
            dtype=out_dtype, 
            compressor=comp, 
            dimension_separator='/')#,
            # fill_value=0,
            # exact=True)

        out_slices = slices_from_chunks(normalize_chunks(dest_arr.chunks, shape=dest_arr.shape))
        # break the slices up into batches, to make things easier for the dask scheduler
        out_slices_partitioned = tuple(partition_all(100000, out_slices))
        for idx, part in enumerate(out_slices_partitioned):
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: save_chunk_zarr(src_arr, dest_arr, v, invert), part)
            print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
            # wait for all the futures to complete
            result = wait(fut)
            print(f'Completed {len(part)} tasks in {time.time() - start}s')


@click.command()
@click.option('--src','-s', type=click.Path(exists = True),help='Input .zarr file location.')
@click.option('--dest', '-d', type=click.STRING, help='Output zarr path' )
@click.option('--workers','-w',default=100,type=click.INT,help = "Number of dask workers")
@click.option('--cluster_type', '-ct', default='' ,type=click.STRING, help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
@click.option('--out_dtype', '-odt', default='', type=click.STRING, help="Output array data type")
@click.option('--compressor', '-c', default='', type=click.STRING, help="Which compression algorithm to use. Options: gzip, zstd" )
def cli(src, dest, workers, cluster_type, out_dtype, compressor):
    
    if cluster_type == '':
        print('Did not specify which instance of the dask client to use!')
        sys.exit(0)
    elif cluster_type == 'lsf':
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
    
    elif cluster_type == 'local':
            cluster = LocalCluster()
    
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)
    
    if compressor == 'zstd':
        comp = Zstd(level=6)
    # if compressor == 'gzip':
    #     comp = Gzip(level=6)


    src_store = zarr.DirectoryStore(src)
    zarr_src = zarr.open(store=src_store, mode = 'r')

    dest_store = zarr.NestedDirectoryStore(dest)
    dest_root = zarr.open_group(store=dest_store, mode= 'a')
    
    copy_arrays(z_src=zarr_src, dest_root=dest_root, client=client, num_workers=workers, comp=comp, invert=False, out_dtype=out_dtype)




if __name__ == '__main__':
    cli()