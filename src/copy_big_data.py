import os
import zarr
import dask.array as da

import dask
from dask.distributed import Client
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster
from dask.array.core import slices_from_chunks

import time 
import numpy as np


def copy_arrays(zarr_arrays, zarrdest, max_dask_chunk_num, invert_contrast, chunks, comp):#client, chunks, num_workers, comp):
    #breakpoint()
    new_store = zarr.NestedDirectoryStore(zarrdest)
    for item in zarr_arrays:
        #old_arr = item
        darray = da.from_array(item, chunks = optimal_dask_chunksize(item, chunks, max_dask_chunk_num, 2))#.astype('int16')
        dataset = zarr.create(store = new_store, path= item.path, shape=item.shape, chunks=item.chunks, dtype=item.dtype, compressor=comp)
        #darray = da.from_array(dataset, chunks=dataset.chunks)
        start_time = time.time()
        
        slab_size = (212, -1, -1)
        slices_src = slices_from_chunks(darray.rechunk(slab_size).chunks)
        for sl_src in slices_src:
            #client.cluster.scale(num_workers)
            
            darr_slc = darray[sl_src]
            optimal_slab_chunks = optimal_dask_chunksize(darr_slc, chunks, max_dask_chunk_num, 2)
            
            temp_darr = da.rechunk(darr_slc, chunks = optimal_slab_chunks)
            print(f'temp arr chunks {temp_darr.chunks}')
            #client.compute(da.store(temp_darr, dataset, regions=sl_src, lock=False), sync=True)
            da.store(temp_darr, dataset, regions=sl_src, lock=False)
                
            #client.cluster.scale(0)

        # if invert_contrast:
        #     invert_darray = da.invert(darray)
        #     da.store(invert_darray, dataset, lock = False)
        # else:
        #     da.store(darray, dataset, lock = False)
        copy_time = time.time() - start_time
        print(f"({copy_time}s) copied {item.name} to {item.path}")


def cluster_compute(scheduler, num_cores):
    def decorator(function):
        def wrapper(*args, **kwargs):
            #dask.config.set({'temporary_directory': '/scratch/zubovy/'})

            if scheduler == "lsf":
                #num_cores = 60
                cluster = LSFCluster( cores=num_cores,
                        processes=1,
                        memory=f"{15 * num_cores}GB",
                        ncpus=num_cores,
                        mem=15 * num_cores,
                        walltime="48:00",
                        death_timeout = 240.0,
                        local_directory = "/scratch/zubovy/"
                        )
                cluster.scale(86)
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

# # calculate automatically what chunk size scaling we should have in order to avoid having a complex dask computation graph. 
# def optimal_dask_chunksize(arr, max_dask_chunk_num):
#     #calculate number of chunks within a zarr array.
#     if isinstance(arr, zarr.core.Array):
#         chunk_dims = arr.chunks
#     else:
#         chunk_dims = arr.chunksize
#     chunk_num= np.prod(arr.shape)/np.prod(chunk_dims) 
    
#     # 1. Scale up chunk size (chunksize approx = 1GB)
#     scaling = 1
#     while np.prod(chunk_dims)*arr.itemsize*pow(scaling, 3)/pow(10, 6) < 300 :
#         scaling += 1

#     # 3. Number of chunks should be < 50000
#     while (chunk_num / pow(scaling,3)) > max_dask_chunk_num:
#         scaling +=1

#     # 2. Make sure that chunk dims < array dims
#     while any([ch_dim > 3*arr_dim/4 for ch_dim, arr_dim in zip(tuple(dim * scaling for dim in chunk_dims), arr.shape)]):#np.prod(chunks)*arr.itemsize*pow(scaling,3) > arr.nbytes:
#         scaling -=1

#     if scaling == 0:
#         scaling = 1

#     return tuple(dim * scaling for dim in chunk_dims) 

def optimal_dask_chunksize(arr, target_chunks, max_dask_chunk_num, scale_dim=3):
    #calculate number of chunks within a zarr array.
    chunk_dims = target_chunks
    chunk_num= np.prod(arr.shape)/np.prod(chunk_dims) 
    
    # 1. Scale up chunk size (chunksize approx = 1GB)
    scaling = 1
    while np.prod(chunk_dims)*arr.itemsize*pow(scaling, 3)/pow(10, 6) < 700 :
        scaling += 1
    
    print(scaling)

    # 2. Number of chunks should be < 50000
    while (chunk_num / pow(scaling,3)) > max_dask_chunk_num:
        scaling +=1
        
    print(scaling)

    # 3. Make sure that chunk dims < array dims
    while any([ch_dim > 3*arr_dim/4 for ch_dim, arr_dim in zip(tuple(dim * scaling for dim in chunk_dims[-scale_dim:]), arr.shape[-scale_dim:])]):#np.prod(chunks)*arr.itemsize*pow(scaling,3) > arr.nbytes:
        scaling -=1
        
    print(scaling)

    if scaling == 0:
        scaling = 1
            
    #anisotropic scaling
    scaling_dims = np.ones(len(chunk_dims), dtype=int)
    
    scaling_dims[-scale_dim:] = scaling
    print(scaling_dims)
        
    return tuple(dim * scale_dim for dim, scale_dim  in zip(chunk_dims, scaling_dims)) 



def get_cluster(scheduler, num_cores):
    if scheduler == "lsf":
        #num_cores = 80
        cluster = LSFCluster( cores=num_cores,
                processes=1,
                memory=f"{15 * num_cores}GB",
                ncpus=num_cores,
                mem=15 * num_cores,
                walltime="48:00",
                death_timeout = 240.0,
                local_directory = "/scratch/zubovy/"
                )
        #cluster.scale(num_cores)
    elif scheduler == "local":
        cluster = LocalCluster()

    return cluster
