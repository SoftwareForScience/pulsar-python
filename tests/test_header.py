"""
    header.py unit tests
"""

import unittest
from filterbank.header import read_header, len_header, fil_double_to_angle


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

    def test_len_header(self):
        """
            When calculating header length
            header.py should return the correct length in bytes
        """
        filename = './pspm8.fil'
        correct_header_length = 242
        header_length = len_header(filename)
        self.assertEqual(header_length, correct_header_length)

    def test_fil_double_to_angle(self):
        """
            When calculating the angle
            header.py should return the correct angle
        """
        double_value = 123000.0
        expect_angle = 12.5
        angle = fil_double_to_angle(double_value)
        self.assertAlmostEqual(angle, expect_angle)


if __name__ == '__main__':
    unittest.main()
