"""
    pipeline.py unit tests
"""

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
        pipeline.Pipeline(filename)

    def test_row_pipeline(self):
        """
            When running the pipeline as stream,
            expect a file with the time per method
        """
        filename = './pspm8.fil'
        pipeline.Pipeline(filename, as_stream=True)
    

    def test_n_rows_pipeline(self):
        """
            When running the pipeline as stream,
            expect a file with the time per method per chunk
        """
        filename = './pspm8.fil'
        pipeline.Pipeline(filename, as_stream=True, n=10)
