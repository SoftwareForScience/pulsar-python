"""
    header.py unit tests
"""

import unittest
from header import read_header, len_header

class TestHeader(unittest.TestCase):
    """
        Class for testing header.py
    """
    def test_reader(self):
        """
            When reading filterbank header
            header.py should return a dictionary
        """
        filename = './pspm8.fil'
        header_dict = read_header(filename)
        self.assertEqual(header_dict[b'machine_id'], 10)
        self.assertEqual(header_dict[b'telescope_id'], 4)
        self.assertEqual(header_dict[b'data_type'], 1)
        self.assertEqual(header_dict[b'fch1'], 433.968)

    def test_wrong_filetype(self):
        """
            When reading incorrect filtype
            header.py should raise a RuntimeError
        """
        filename = './header.py'
        with self.assertRaises(RuntimeError):
            read_header(filename)

    def test_len_header(self):
        """
            When calculating header length
            header.py should return the correct length in bytes
        """
        filename = './pspm8.fil'
        correct_header_length = 242
        header_length = len_header(filename)
        self.assertEqual(header_length, correct_header_length)

if __name__ == '__main__':
    unittest.main()
