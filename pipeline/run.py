"""
    Script for running the pipeline
"""
import os
import pipeline

for i in range(10):
    # # read static
    pipeline.Pipeline(filename=os.path.abspath("E:/11100335.320.all.fil"), size=49150)
    # read stream, row per row
    pipeline.Pipeline(filename=os.path.abspath("E:/11100335.320.all.fil"), as_stream=True)
    # read stream, n rows
    pipeline.Pipeline(filename=os.path.abspath("E:/11100335.320.all.fil"), as_stream=True, n=10)
