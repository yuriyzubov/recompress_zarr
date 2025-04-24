import os
import zarr
import dask.array as da

from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster
import time 
import numpy as np


def copy_arrays(zarr_arrays, zarrdest, max_dask_chunk_num, comp):
    new_store = zarr.NestedDirectoryStore(zarrdest)
    for item in zarr_arrays:
        old_arr = item[1]
        darray = da.transpose(da.from_array(old_arr, chunks = optimal_dask_chunksize(old_arr, max_dask_chunk_num)))
        dataset = zarr.create(store = new_store, path= old_arr.path+"_transposed", mode='w', shape=old_arr.shape[::-1], chunks=old_arr.chunks[::-1], dtype=old_arr.dtype, compressor=comp)
        
        start_time = time.time()

        da.store(darray, dataset, lock = False)
        copy_time = time.time() - start_time
        print(f"({copy_time}s) copied {old_arr.name} to {old_arr.path}")


def cluster_compute(scheduler, num_cores):
    def decorator(function):
        def wrapper(*args, **kwargs):
            if scheduler == "lsf":
                num_cores = 20
                cluster = LSFCluster( cores=num_cores,
                        processes=1,
                        memory=f"{15 * num_cores}GB",
                        ncpus=num_cores,
                        mem=15 * num_cores,
                        walltime="48:00"
                        )
                cluster.scale(num_cores)
            elif scheduler == "local":
                cluster = LocalCluster()

            with Client(cluster) as cl:
                text_file = open(os.path.join(os.getcwd(), "dask_dashboard_link" + ".txt"), "w")
                text_file.write(str(cl.dashboard_link))
                text_file.close()
                cl.compute(function(*args, **kwargs), sync=True)
        return wrapper
    return decorator

# raise dask warning
def chunk_num_warning(darr):
    chunk_num =  da.true_divide(darr.shape, darr.chunksize).prod()
    if (chunk_num > pow(10, 5)):
        
        log_file_path = os.path.join(os.getcwd(), "warnings")
        os.mkdir(log_file_path)

        warning_file = open(os.path.join(log_file_path, "dask_warning" + ".txt"), "a")
        warning_file.write("Warning: dask array contains more than 100,000 chunks.")
        warning_file.close()  

# calculate automatically what chunk size scaling we should have in order to avoid having a complex dask computation graph. 
def optimal_dask_chunksize(arr, max_dask_chunk_num):
    #calculate number of chunks within a zarr array.
    if isinstance(arr, zarr.core.Array):
        chunk_dims = arr.chunks
    else:
        chunk_dims = arr.chunksize
    chunk_num= np.prod(arr.shape)/np.prod(chunk_dims) 
    
    # 1. Scale up chunk size (chunksize approx = 1GB)
    scaling = 1
    while np.prod(chunk_dims)*arr.itemsize*pow(scaling, 3)/pow(10, 6) < 300 :
        scaling += 1

    # 3. Number of chunks should be < 50000
    while (chunk_num / pow(scaling,3)) > max_dask_chunk_num:
        scaling +=1

    # 2. Make sure that chunk dims < array dims
    while any([ch_dim > 3*arr_dim/4 for ch_dim, arr_dim in zip(tuple(dim * scaling for dim in chunk_dims), arr.shape)]):#np.prod(chunks)*arr.itemsize*pow(scaling,3) > arr.nbytes:
        scaling -=1

    if scaling == 0:
        scaling = 1

    return tuple(dim * scaling for dim in chunk_dims) 
