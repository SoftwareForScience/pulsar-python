"""
    Script for running the pipeline
"""

import pipeline

# read static
pipeline.Pipeline(filename='pspm32.fil')
# read stream, row per row
pipeline.Pipeline(filename='pspm32.fil', as_stream=True)
# read stream, n rows
pipeline.Pipeline(filename='pspm32.fil', as_stream=True, n=10)
