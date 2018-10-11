"""
    filterbank.py unit tests
"""

import unittest
from filterbank import Filterbank

class TestFilterbank(unittest.TestCase):
    """
        Class for testing filterbank.py
    """
    def test_wrong_filename_raises_error(self):
        """
            Initialize filterbank with an incorrect filename
        """
        filename = './thispathdoesnotexist'
        with self.assertRaises(FileNotFoundError):
            Filterbank(filename)

    def test_select_frequency_range(self):
        """
            Initialize filterbank with a correct filename
            and test if values are in frequency range
        """
        filename = './pspm8.fil'
        fil = Filterbank(filename)
        data = fil.select_data(freq_start=431, freq_stop=432)
        self.assertTrue(all(430.5 < i < 432.4  for i in data[0]))

if __name__ == '__main__':
    unittest.main()
