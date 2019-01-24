"""
    pipeline.py unit tests
"""
import os
import unittest

from .context import pipeline

class TestPipeline(unittest.TestCase):
    """
        Class for testing pipeline.py
    """

    def test_static_pipeline(self):
        """
            When running the static pipeline,
            expect a file with the time per method
        """
        filename = './pspm8.fil'
        new_file = './static_filterbank.txt'
        pipeline.Pipeline(filename)
        self.assertTrue(os.path.exists(new_file))
        os.remove(new_file)

    def test_row_pipeline(self):
        """
            When running the pipeline as stream,
            expect a file with the time per method
        """
        filename = './pspm8.fil'
        new_file = './rows_filterbank.txt'
        pipeline.Pipeline(filename, as_stream=True)
        self.assertTrue(os.path.exists(new_file))
        os.remove(new_file)

    def test_n_rows_pipeline(self):
        """
            When running the pipeline as stream,
            expect a file with the time per method per chunk
        """
        filename = './pspm8.fil'
        new_file = './n_rows_filterbank.txt'
        pipeline.Pipeline(filename, as_stream=True, n=10)
        self.assertTrue(os.path.exists(new_file))
        os.remove(new_file)

if __name__ == '__main__':
    unittest.main()
