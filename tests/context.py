"""
    Context import file. This is used by the test files.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# pylint: disable-msg=C0413
import fourier
import filterbank.header as header
import plot
import filterbank.filterbank as filterbank
