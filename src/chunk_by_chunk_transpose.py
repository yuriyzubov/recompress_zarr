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
from numcodecs.gzip import GZip
from toolz import partition_all
import sys

import click


def save_chunk_zarr_transpose(
        source: zarr.Array,
        dest: zarr.Array,
        out_slices: Tuple[slice, ...],
        transpose_axes: Tuple[int, ...],
        invert: bool):

    # Read from source with the original slice coordinates
    in_slices = tuple(out_slice for out_slice in out_slices)
    source_data = source[in_slices]

    # only store source_data if it is not all 0s
    if not (source_data == 0).all():
        # Transpose the data
        transposed_data = np.transpose(source_data, axes=transpose_axes)

        # Transpose the slice coordinates to match the transposed dimensions
        transposed_slices = tuple(out_slices[i] for i in transpose_axes)

        if invert == True:
            dest[transposed_slices] = np.invert(transposed_data)
        else:
            dest[transposed_slices] = transposed_data
    return 1

def transpose_arrays(z_src: zarr.Group | zarr.Array,
                dest_root: zarr.Group,
                client: Client,
                num_workers: int,
                comp: Codec,
                invert: bool,
                out_dtype: str,
                out_chunksize : list[int],
                transpose_axes: Tuple[int, ...]):


    # store original array in a new .zarr file as an arr_name
    client.cluster.scale(num_workers)
    if isinstance(z_src, zarr.core.Array):
        # If source is an array at root level, extract name from store path
        if not z_src.name:
            z_src._name = os.path.basename(z_src.store.path.rstrip('/'))
        z_arrays = [z_src]
    else:
        z_arrays = [key_val_arr[1] for key_val_arr in (z_src.arrays())]

    for src_arr in z_arrays:

        if out_dtype=='':
            out_dtype = src_arr.dtype

        if comp==None:
            comp = src_arr.compressor

        # Transpose the shape and chunks according to transpose_axes
        transposed_shape = tuple(src_arr.shape[i] for i in transpose_axes)

        if out_chunksize==None:
            transposed_chunks = tuple(src_arr.chunks[i] for i in transpose_axes)
        else:
            transposed_chunks = out_chunksize

        start_time = time.time()
        dest_arr = dest_root.require_dataset(
            src_arr.name,
            shape=transposed_shape,
            chunks=transposed_chunks,
            dtype=out_dtype,
            compressor=comp,
            dimension_separator='/',
            fill_value=0)

        # Use the original (non-transposed) chunks for reading
        out_slices = slices_from_chunks(normalize_chunks(src_arr.chunks, shape=src_arr.shape))
        # break the slices up into batches, to make things easier for the dask scheduler
        out_slices_partitioned = tuple(partition_all(100000, out_slices))
        for idx, part in enumerate(out_slices_partitioned):
            print(f'{idx + 1} / {len(out_slices_partitioned)}')
            start = time.time()
            fut = client.map(lambda v: save_chunk_zarr_transpose(src_arr, dest_arr, v, transpose_axes, invert), part)
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
@click.option('--invert', '-i', default=False, type=click.BOOL, help = 'invert values of the array when writing into zarr. Default: false')
@click.option('--out_chunksize', nargs=3, default=None, type=click.INT, help= 'specify output chunksize')
@click.option('--transpose_axes', '-t', nargs=3, default=(2, 1, 0), type=click.INT, help= 'specify transpose axes order (default: 2,1,0 for reversing 3D)')
@click.option('--project_name', '-p' , default=None, type=click.STRING, help= 'specify project name')

def cli(src,
        dest,
        workers,
        cluster_type,
        out_dtype,
        compressor,
        invert,
        out_chunksize,
        transpose_axes,
        project_name):

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
            local_directory = "/scratch/$USER/",
            job_extra_directives = [f'-P {project_name}']
            )

    elif cluster_type == 'local':
            cluster = LocalCluster()

    client = Client(cluster)
    with open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w") as text_file:
        text_file.write(str(client.dashboard_link))
    print(client.dashboard_link)

    comp = None
    if compressor == 'zstd':
        comp = Zstd(level=6)
    elif compressor == 'gzip':
        comp = GZip(level=6)


    src_store = zarr.DirectoryStore(src)
    zarr_src = zarr.open(store=src_store, mode = 'r')

    dest_store = zarr.NestedDirectoryStore(dest)
    dest_root = zarr.open_group(store=dest_store, mode= 'a')

    transpose_arrays(z_src=zarr_src,
                dest_root=dest_root,
                client=client,
                num_workers=workers,
                comp=comp,
                invert=False,
                out_dtype=out_dtype,
                out_chunksize=out_chunksize,
                transpose_axes=transpose_axes)

if __name__ == '__main__':
    cli()
