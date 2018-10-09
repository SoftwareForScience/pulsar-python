"""
Reader.py unit tests
"""

import unittest
import reader

class TestCase(unittest.TestCase):
    """
    Class TestCase
    """
    def test_reader(self):
        """
        Perform reader header test
        """
        filename = "./pspm_tiny.fil"
        header_dict = reader.read_header(filename)
        self.assertEqual(header_dict[b'machine_id'], 10)
        self.assertEqual(header_dict[b'telescope_id'], 4)
        self.assertEqual(header_dict[b'data_type'], 1)
        self.assertEqual(header_dict[b'fch1'], 433.968)

if __name__ == '__main__':
    unittest.main()
