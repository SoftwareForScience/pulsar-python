"""
    Script for running the pipeline
"""
import os
import pipeline

fil_name = os.path.abspath("E:/11100335.320.all.fil")

for i in range(1000):
    # read static
    pipeline.Pipeline(filename=fil_name, size=49150)
    # read stream, row per row
    pipeline.Pipeline(filename=fil_name, as_stream=True)
    # read stream, n rows
    pipeline.Pipeline(filename=fil_name, as_stream=True, n=10)
