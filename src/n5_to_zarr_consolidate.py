import os
from typing import Tuple, List
from dask.distributed import wait
from dask_jobqueue import LSFCluster
from dask.distributed import LocalCluster
import natsort
import pint
from operator import itemgetter
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

def apply_ome_template(zgroup : zarr.Group, new_axis_type : str):
    
        z_attrs = {"multiscales": [{}]}

        # normalize input units, i.e. 'meter' or 'm'-> 'meter'
        ureg = pint.UnitRegistry()
        units_list = [ureg.Unit(unit) for unit in zgroup.attrs['units']]            

        if new_axis_type == 'time':
            consolidated_axes = [{"name" : 't', 'type' : 'time', 'unit' : 'seconds'}]
        elif new_axis_type == 'channel':
            consolidated_axes = [{"name" : 'c', 'type' : 'channel'}]
        #populate .zattrs
        sc = [1.0]
        sc.extend([1.0,]*len(units_list))
        
        consolidated_axes.extend([{"name": axis, 
                                            "type": "space",
                                            "unit": str(unit)} for (axis, unit) in zip(zgroup.attrs['axes'], 
                                                                                    units_list)])
        z_attrs['multiscales'][0]['axes'] = consolidated_axes
        z_attrs['multiscales'][0]['version'] = '0.4'
        z_attrs['multiscales'][0]['name'] = zgroup.name
        z_attrs['multiscales'][0]['coordinateTransformations'] = [{"type": "scale",
                        "scale": sc}
                        ]
        
        return z_attrs

def ome_dataset_metadata(n5arr : zarr.Array,
                            group : zarr.Group):

        arr_attrs_n5 = n5arr.attrs['transform']
        sc = [1.0]
        sc.extend(arr_attrs_n5['scale'])
        tr = [0.0]
        tr.extend(arr_attrs_n5['translate'])
        
        dataset_meta =  {
                        "path": os.path.relpath(n5arr.path, group.path),
                        "coordinateTransformations": [{
                            'type': 'scale',
                            'scale': sc},{
                            'type': 'translation',
                            'translation' : tr
                        }]}
    
        return dataset_meta

def ome_metadata_consolidated(src_zgroup : zarr.Group, dest_group : zarr.Group, new_axis_type : str):
    group_keys = list(src_zgroup.keys())
    key = group_keys[0]
    if isinstance(src_zgroup[key], zarr.hierarchy.Group):
        # if key!='/':
        #     ome_metadata_consolidated(src_zgroup[key], dest_group, new_axis_type)
        if 'scales' in src_zgroup[key].attrs.asdict():
            zattrs = apply_ome_template(src_zgroup[key], new_axis_type=new_axis_type)
            zarrays = src_zgroup[key].arrays()

            unsorted_datasets = []
            for arr in zarrays:
                unsorted_datasets.append(ome_dataset_metadata(arr[1], src_zgroup[key]))

            #1.apply natural sort to organize datasets metadata array for different resolution degrees (s0 -> s10)
            #2.add datasets metadata to the omengff template
            zattrs['multiscales'][0]['datasets'] = natsort.natsorted(unsorted_datasets, key=itemgetter(*['path']))
            dest_group.attrs['multiscales'] = zattrs['multiscales']
                    

def consolidate_chunk_zarr(
        src_arrs: List[zarr.Array], 
        dest: zarr.Array, 
        out_slices: Tuple[slice, ...],
        invert: bool):
    
    in_slices = tuple(out_slice for out_slice in out_slices)
    src_data_stacked = np.stack([arr[in_slices[1:]] for arr in src_arrs], axis=0) 
    # only store source_data if it is not all 0s
    if not (src_data_stacked == 0).all():
        if invert == True:
            dest[out_slices] = np.invert(src_data_stacked)
        else:
            dest[out_slices] = src_data_stacked
    return 1

def consolidate_arrays(src_arrs: List[zarr.Array],
                dest_root: zarr.Group,
                dest_arr_name : str,
                client: Client,
                comp: Codec,
                invert: bool,
                out_dtype: str,
                out_chunksize : list[int]):
    
    # store original arrays in a new .zarr file as a dest_arr_name            
    if out_dtype==None:
        out_dtype = src_arrs[0].dtype
      
    if comp==None:
        n5_compressor = src_arrs[0].compressor
        if hasattr(n5_compressor, 'compressor_config') and n5_compressor.compressor_config is not None:
            from numcodecs import get_codec
            comp = get_codec(n5_compressor.compressor_config)

        
    if out_chunksize==None:
        out_chunksize=[len(src_arrs), *src_arrs[0].chunks]

    start_time = time.time()
    dest_arr = dest_root.require_dataset(
        dest_arr_name, 
        shape=[len(src_arrs), *src_arrs[0].shape], 
        chunks=out_chunksize, 
        dtype=out_dtype, 
        compressor=comp, 
        dimension_separator='/',
        fill_value=0)#np.array(0, dtype=out_dtype))
        # exact=True)

    out_slices = slices_from_chunks(normalize_chunks(dest_arr.chunks, shape=dest_arr.shape))
    #break the slices up into batches, to make things easier for the dask scheduler
    out_slices_partitioned = tuple(partition_all(100000, out_slices))
    for idx, part in enumerate(out_slices_partitioned):
        print(f'{idx + 1} / {len(out_slices_partitioned)}')
        start = time.time()
        fut = client.map(lambda v: consolidate_chunk_zarr(src_arrs, dest_arr, v, invert), part)
        print(f'Submitted {len(part)} tasks to the scheduler in {time.time()- start}s')
        # wait for all the futures to complete
        result = wait(fut)
        print(f'Completed {len(part)} tasks in {time.time() - start}s')


@click.command()
@click.option('--src','-s', type=click.Path(exists = True),help='Input .zarr file location.')
@click.option('--dest', '-d', type=click.STRING, help='Output zarr path' )
@click.option('--workers','-w',default=100,type=click.INT,help = "Number of dask workers")
@click.option('--cluster_type', '-ct', default='' ,type=click.STRING, help="Which instance of dask client to use. Local client - 'local', cluster 'lsf'")
@click.option('--out_dtype', '-odt', default=None, type=click.STRING, help="Output array data type")
@click.option('--compressor', '-c', default=None, type=click.STRING, help="Which compression algorithm to use. Options: gzip, zstd" )
@click.option('--invert', '-i', default=False, type=click.BOOL, help = 'invert values of the array when writing into zarr. Default: false')
@click.option('--out_chunksize', nargs=3, default=None, type=click.INT, help= 'specify output chunksize')
@click.option('--new_axis_type', default='channel', type=click.STRING, help= 'type of the dimension that is being consolidated. Options: time, channel')
@click.option('--dask_log_dir', default=None, type=click.STRING, help= 'specify dask log directory')
def cli(src,
        dest,
        workers,
        cluster_type,
        out_dtype,
        compressor,
        invert,
        out_chunksize, 
        new_axis_type, 
        dask_log_dir):
    
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
            log_directory = dask_log_dir,
            local_directory = "/scratch/$USER/"
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
    if compressor == 'gzip':
        comp = GZip(level=6)


    src_store = zarr.N5Store(src)
    src_root = zarr.open(store=src_store, mode = 'r')
    
    xtra_dim_groups = list(src_root.groups())
    dest_store = zarr.NestedDirectoryStore(dest)
    dest_root = zarr.open_group(store=dest_store, mode= 'a')
        
    arr_shapes = []
    for dim in xtra_dim_groups:
        dim_arrs = dim[1].arrays()
        dim_arr_shapes = {arr[0] : arr[1].shape for arr in dim_arrs}
        arr_shapes.append(dim_arr_shapes)
        
    arr_shapes_by_level = {k: [d[k] for d in arr_shapes] for k in arr_shapes[0]}
    
    # check that for every multiscale level, array shape is the same across all channels
    for arr_name in arr_shapes_by_level:
        if len(set(arr_shapes_by_level[arr_name])) > 1:
            raise ValueError(f'Array shape is inconsistent for array {arr_name}')
    
    # create metadata
    ome_metadata_consolidated(src_root, dest_root, new_axis_type=new_axis_type)
    # consolidate arrays into one
    client.cluster.scale(workers)
    for level_arr in arr_shapes_by_level:
        level_arrays = [dim_group[1][level_arr] for dim_group in list(xtra_dim_groups)]
        consolidate_arrays(src_arrs=level_arrays,
                    dest_root=dest_root,
                    dest_arr_name = level_arr,
                    client=client,
                    comp=comp,
                    invert=False,
                    out_dtype=out_dtype,
                    out_chunksize=out_chunksize)
        
    client.cluster.scale(0)

if __name__ == '__main__':
    cli()