"""
    clipping.py unit tests
"""

import unittest
from .context import clipping # pylint: disable-msg=E0611

class TestClipping(unittest.TestCase):
    """
        Class for testing clipping.py
    """

    fil_vector = [[1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [5, 5, 5, 5, 5, 5, 5, 5],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1],
                  [1, 1, 1, 1, 5, 1, 1, 1]]

    def test_filter_samples(self):
        """
            When filtering samples all time samples
            with a relative high power should be removed
        """
        result = clipping.filter_samples(self.fil_vector)
        expect = self.fil_vector
        del expect[2]
        self.assertListEqual(result, expect)
