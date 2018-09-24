import unittest
from helloworld import HelloWorld

class testCase (unittest.TestCase):
    def test_hello_world(self):
        helloworld = HelloWorld()
        self.assertEqual(helloworld.message, "Hello World")

if __name__ == '__main__':
    unittest.main()