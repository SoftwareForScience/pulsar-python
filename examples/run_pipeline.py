"""
    Script for running the pipeline
"""
#pylint: disable-all
import os,sys,inspect
CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)
from pipeline.pipeline import Pipeline

# init filterbank filename
fil_name = "./pspm.fil"
# init filterbank sample size
sample_size = 49152
# init times the pipeline should run
n_times = 10

# run the filterbank n times
for i in range(n_times):
    # read static
    Pipeline(filename=fil_name, size=sample_size)
    # read stream, row per row
    Pipeline(filename=fil_name, as_stream=True)
    # read stream, n rows
    Pipeline(filename=fil_name, as_stream=True, n=sample_size)
