"""
Define test case
"""
import unittest
import helloworld

class TestCase(unittest.TestCase):
    """
    Define test case
    """
    def test_hello_world(self):
        """
        Perform test
        """
        self.assertEqual(helloworld.hello_world(), "Hello World!")

if __name__ == '__main__':
    unittest.main()
