"""
    filterbank.py tests
"""

import unittest
from .context import filterbank # pylint: disable-msg=E0611

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
            filterbank.Filterbank(filename)

    def test_select_frequency_range(self):
        """
            Initialize 8 bits filterbank
            and test if values are in frequency range
        """
        filename = './pspm8.fil'
        fil = filterbank.Filterbank(filename)
        fil.read_filterbank()
        data = fil.select_data(freq_start=431, freq_stop=432)
        self.assertTrue(all(430.5 < i < 432.4  for i in data[0]))

    def test_select_frequency_range_reversed(self):
        """
            Initialize 16 bits filterbank
            and test if values are in frequency range
        """
        filename = './pspm16.fil'
        fil = filterbank.Filterbank(filename)
        fil.read_filterbank()
        data = fil.select_data(freq_start=432, freq_stop=431)
        self.assertTrue(all(430.5 < i < 432.4  for i in data[0]))

    def test_filterbank_time_range(self):
        """
            Initialize 8 bits filterbank
            and test if values are in time range
        """
        filename = './pspm8.fil'
        time_range = (10, 30)
        time_delt = abs(time_range[0] - time_range[1])
        fil = filterbank.Filterbank(filename)
        fil.read_filterbank()
        data = fil.select_data(time_start=time_range[0], time_stop=time_range[1])
        self.assertEqual(len(data[1]), time_delt)

    def test_filterbank_parameters(self):
        """
            Initialize 32 bits filterbank
            and test if all parameters work
        """
        filename = './pspm32.fil'
        freq_range = (433, 435)
        time_range = (10, 20)
        time_delt = abs(time_range[0] - time_range[1])
        fil = filterbank.Filterbank(filename, freq_range=freq_range, time_range=time_range)
        fil.read_filterbank()
        data = fil.select_data()
        self.assertTrue(all(432.5 < i < 435.4  for i in data[0]))
        self.assertEqual(len(data[1]), time_delt)

    def test_filterbank_rows(self):
        """
            Read a single row as stream
            and test if it returns a row
        """
        filename = './pspm32.fil'
        fil = filterbank.Filterbank(filename)
        data = fil.next_row()
        self.assertGreater(len(data), 0)

    def test_filterbank_rows_empty(self):
        """
            Read stream till end of file
            and test if it returns True at the end
        """
        filename = './pspm32.fil'
        fil = filterbank.Filterbank(filename)
        # read rows till method returns Boolean
        while not isinstance(fil.next_row(), bool):
            pass
        self.assertTrue(fil.next_row())

    def test_filterbank_n_rows(self):
        """
            Read n rows from stream
            and test if it returns n rows
        """
        n_rows = 10
        filename = './pspm32.fil'
        fil = filterbank.Filterbank(filename)
        data = fil.next_n_rows(n_rows)
        self.assertEqual(len(data), n_rows)

    def test_filterbank_n_rows_empty(self):
        """
            Read n rows from stream till end of file
            and test if it returns True at the end
        """
        n_rows = 10
        filename = './pspm32.fil'
        fil = filterbank.Filterbank(filename)
        # read n rows till method returns Boolean
        while not isinstance(fil.next_n_rows(n_rows)):
            pass
        self.assertTrue(fil.next_n_rows(n_rows))

if __name__ == '__main__':
    unittest.main()
