"""
Define test case
"""
import unittest
from helloworld import HelloWorld

class TestCase(unittest.TestCase):
    """
    Define test case
    """
    def test_hello_world(self):
        """
        Perform test
        """
        helloworld = HelloWorld()
        self.assertEqual(helloworld.message, "Hello World")

if __name__ == '__main__':
    unittest.main()
