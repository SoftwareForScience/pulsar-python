import os
import sys
import inspect

CURRENT_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

from filterbank.filterbank import Filterbank

fb = Filterbank(filename='examples/pspm8.fil', stream=True)

# read first 10 rows
# for i in range(10):
    # print(fb.next_row())
# print(fb.next_row())

fb2 = Filterbank(filename='examples/pspm8.fil')

fil_data = fb2.select_data()

print(fil_data[0].size)
print(fil_data[1].size)