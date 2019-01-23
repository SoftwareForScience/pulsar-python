"""
    generate.py tests
"""
import os
import unittest
from .context import generate # pylint: disable-msg=E0611
from .context import filterbank

class TestGenerate(unittest.TestCase):
    """
        Class for testing generate.py
    """

    header_dict = {
        b'source_name': b'P: 80.0000 ms, DM: 200.000',
        b'telescope_id': 0,
        b'machine_id': 0,
        b'tstart': 0,
        b'foff': -0.062,
        b'nchans': 128,
        b'tsamp': 8e-05,
        b'period': .5,
        b'fch1': 400,
        b'nifs': 1
    }

    def test_generate_8bit_fil(self):
        """
            Generate a fake 8bit filterbank file
            and remove afterwards
        """
        filename = './pspm_test.fil'
        self.header_dict[b'nbits'] = 8
        generate.generate_file(filename, self.header_dict)
        fil = filterbank.Filterbank(filename, read_all=True)
        _, fil_data = fil.select_data()
        self.assertEqual(self.header_dict[b'telescope_id'], fil.header[b'telescope_id'])
        self.assertEqual(self.header_dict[b'machine_id'], fil.header[b'machine_id'])
        self.assertEqual(self.header_dict[b'nbits'], fil.header[b'nbits'])
        self.assertEqual(self.header_dict[b'nchans'], len(fil_data[0]))
        os.remove(filename)

    def test_generate_16bit_fil(self):
        """
            Generate a fake 16bit filterbank file
            and remove afterwards
        """
        filename = './pspm_test.fil'
        self.header_dict[b'nbits'] = 16
        generate.generate_file(filename, self.header_dict)
        fil = filterbank.Filterbank(filename, read_all=True)
        _, fil_data = fil.select_data()
        self.assertEqual(self.header_dict[b'telescope_id'], fil.header[b'telescope_id'])
        self.assertEqual(self.header_dict[b'machine_id'], fil.header[b'machine_id'])
        self.assertEqual(self.header_dict[b'nbits'], fil.header[b'nbits'])
        self.assertEqual(self.header_dict[b'nchans'], len(fil_data[0]))
        os.remove(filename)

    def test_generate_32bit_fil(self):
        """
            Generate a fake 32bit filterbank file
            and remove afterwards
        """
        filename = './pspm_test.fil'
        self.header_dict[b'nbits'] = 32
        generate.generate_file(filename, self.header_dict)
        fil = filterbank.Filterbank(filename, read_all=True)
        _, fil_data = fil.select_data()
        self.assertEqual(self.header_dict[b'telescope_id'], fil.header[b'telescope_id'])
        self.assertEqual(self.header_dict[b'machine_id'], fil.header[b'machine_id'])
        self.assertEqual(self.header_dict[b'nbits'], fil.header[b'nbits'])
        self.assertEqual(self.header_dict[b'nchans'], len(fil_data[0]))
        os.remove(filename)

if __name__ == '__main__':
    unittest.main()
