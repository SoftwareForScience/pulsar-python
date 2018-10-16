"""
    Context import file. This is used by the test files.
"""
# pylint: disable-msg=C0414
# pylint: disable-msg=W0611
# pylint: disable-msg=C0412
# pylint: disable-msg=C0413
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fourier
import filterbank.header as header
import plot
import filterbank.filterbank as filterbank
