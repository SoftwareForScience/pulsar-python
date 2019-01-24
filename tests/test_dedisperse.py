"""
    dedisperse.py unit tests
"""

import unittest
import numpy as np

from .context import dedisperse # pylint: disable-msg=E0611

class TestDedisperse(unittest.TestCase):
    """
        Class for testing dedisperse.py
    """

    samples = [[10, 1, 1, 1, 1, 1, 1],
               [1, 10, 1, 1, 1, 1, 1],
               [1, 1, 10, 1, 1, 1, 1],
               [1, 1, 1, 10, 1, 1, 1],
               [1, 1, 1, 1, 10, 1, 1],
               [1, 1, 1, 1, 1, 10, 1],
               [1, 1, 1, 1, 1, 1, 10]]

    def test_dedisperse(self):
        """
            When performining dedispersion,
            expect moved frequencies per sample
        """
        disp_measure = 6
        results = dedisperse.dedisperse(np.array(self.samples), None, None, disp_measure)
        self.assertListEqual(list(results[len(self.samples)-1]), [10]*7)
