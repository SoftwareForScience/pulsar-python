import reader
import unittest

class TestCase(unittest.TestCase):
    
    def test_reader(self):
        filename = "./pspm_tiny.fil"
        header_dict = reader.read_header(filename)#{b'source_name': b'P: 3.141592700000 ms, DM: 1.000', b'machine_id': 10, b'telescope_id': 4, b'data_type': 1, b'fch1': 433.968, b'foff': -0.062, b'nchans': 128, b'nbits': 1, b'tstart': 50000.0, b'tsamp': 8e-05, b'nifs': 1}
        self.assertEqual(header_dict[b'machine_id'], 10)
        self.assertEqual(header_dict[b'telescope_id'], 4)
        self.assertEqual(header_dict[b'data_type'], 1)
        self.assertEqual(header_dict[b'fch1'], 433.968)

if __name__ == '__main__':
    unittest.main()