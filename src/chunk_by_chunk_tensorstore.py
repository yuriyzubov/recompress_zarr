import json
import os
from typing import Literal, Tuple, Union
from dask.distributed import wait
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster


import dask.array as da
import zarr
import time
from numcodecs.abc import Codec
import numpy as np
from dask.array.core import slices_from_chunks, normalize_chunks
from dask.distributed import Client
from numcodecs import Zstd
from numcodecs import GZip
from toolz import partition_all

import tensorstore as ts
import click

def open_ds_tensorstore(dataset_path: str,  driver : str,  mode="r"):
    spec = {
        "driver": driver,
        "kvstore": {
            "driver": "file",
            "path": dataset_path
            },
    }
    if mode == "r":
        dataset_future = ts.open(spec, read=True, write=False)
    else:
        dataset_future = ts.open(spec, read=False, write=True)

    return dataset_future.result()
    
def save_chunk_ts(
        source: ts.TensorStore, 
        dest: zarr.Array,#ts.TensorStore, 
        out_slices: Tuple[slice, ...],
        invert: bool):
    
    in_slices = tuple(out_slice for out_slice in out_slices)
    source_data = np.array(source[in_slices])  
    # only store source_data if it is not all 0s
    if not (source_data == 0).all():
        if invert == True:
            dest[out_slices] = np.invert(source_data)
        else:
            dest[out_slices] = source_data
    return 1

def copy_arrays(ts_src: ts.TensorStore,
                dest_root: zarr.Group,
                client: Client,
                num_workers: int,
                dest_comp: Codec,
                invert: bool,
                dest_dtype: str):
    
    
    # store original tensorstore array in a new .zarr file as an arr_name
    client.cluster.scale(num_workers)

            
    if dest_dtype=='':
        dest_dtype = ts_src.dtype.name
        
    start_time = time.time()
    dest_arr = dest_root.require_dataset(
        os.path.basename(os.path.normpath(ts_src.kvstore.path)), 
        shape=ts_src.shape, 
        chunks=tuple(ts_src.chunk_layout.to_json()['read_chunk']['shape']), 
        dtype=dest_dtype, 
        compressor=dest_comp, 
        dimension_separator='/')
        
    out_slices = slices_from_chunks(normalize_chunks(dest_arr.chunks, shape=dest_arr.shape))
    # break the slices up into batches, to make things easier for the dask scheduler
    out_slices_partitioned = tuple(partition_all(100000, out_slices))
    for idx, part in enumerate(out_slices_partitioned):
        print(f'{idx + 1} / {len(out_slices_partitioned)}')
        start = time.time()
        fut = client.map(lambda v: save_chunk_ts(ts_src, dest_arr, v, invert), part)
        print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
        # wait for all the futures to complete
        result = wait(fut)
        print(f'Completed {len(part)} tasks in {time.time() - start}s')

@click.command()
@click.option('--src', type=click.STRING, help='Input tensorstore array path.')
@click.option('--src_driver', default="", type=click.STRING, 'tensor store driver, (zarr, neuroglancer_precomputed, n5)' )
@click.option('--dest', type=click.STRING,  help='Output .zarr file location.')
@click.option('--dest_compressor', '-dc', default="", type=click.STRING, )
@click.option('--scheduler', '-s', default='lsf' ,type=click.STRING, help = 'dask scheduler. "lsf"(LSF Cluster) or "local" (Single machine)')
@click.option('--num_workers', '-w', default = 20, type=click.INT, help='Number of dask workers. Default = 20.')
@click.option('--dest_dtype', '-ddt', default = '', type=click.STRING, 'output zarr array dtype. Input array dtype is used as a default type')
@click.option('--invert', '-i', default=False, type=click.BOOL, help = 'invert values of the array when writing into zarr. Default: false')
def cli(src, src_driver, dest, dest_compressor, scheduler, num_workers, dest_dtype, invert):
    if dest_compressor=='zstd':
        comp = Zstd(level=6)
    elif dest_compressor=='gzip':
        comp = GZip(level=6)
    else:
        raise ValueError("No compression method specified")
 
    num_cores = 1
    if scheduler=='lsf':
        cluster = LSFCluster(
            cores=num_cores,
            processes=num_cores,
            memory=f"{15 * num_cores}GB",
            ncpus=num_cores,
            mem=15 * num_cores,
            walltime="48:00",
            local_directory = "/scratch/$USER/"
            )
    elif scheduler=='local': 
        cluster = LocalCluster()
        
    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)


    if src_driver == 'neuroglancer_precomputed': 
        src_arr = open_ds_tensorstore(src, driver=src_driver, mode='r')[ts.d['channel'][0]]
    else: 
        src_arr = open_ds_tensorstore(src, driver=src_driver, mode='r')
        
    dest_store = zarr.NestedDirectoryStore(dest)
    dest_root = zarr.open_group(store=dest_store, mode= 'a')
    
    copy_arrays(ts_src=src_arr, dest_root=dest_root, client=client, num_workers=num_workers, dest_comp=comp, invert=invert, dest_dtype = dest_dtype)
    

if __name__ == '__main__':
    
    cli()