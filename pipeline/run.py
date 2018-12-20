"""
    Script for running the pipeline
"""
import os
import pipeline

# read static
# pipeline.Pipeline(filename=os.path.abspath("E:/11100335.320.all.fil"))
# read stream, row per row
# pipeline.Pipeline(filename=os.path.abspath("E:/11100335.320.all.fil"), as_stream=True)
# read stream, n rows
pipeline.Pipeline(filename=os.path.abspath("E:/11100335.320.all.fil"), as_stream=True, n=10)
