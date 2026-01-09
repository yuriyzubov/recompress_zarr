This script could be used for recompressing, fast copy, and store change of zarr arrays. 
#### How to run
1. open command line terminal
2. install poetry tool for dependency management and packaging: https://pypi.org/project/poetry/
3. switch to the recompress_zarr/src directory:
    ``cd PATH_TO_DIRECTORY/recompress_zarr/src``
4. install python dependencies:
    ``poetry install``
5. run script using cli:
    ``poetry run python chunk_by_chunk_copy.py "PATH_TO_SOURCE_DIRECTORY/input_file.zarr" "PATH_TO_DEST_DIRECTORY/output_file.zarr"``