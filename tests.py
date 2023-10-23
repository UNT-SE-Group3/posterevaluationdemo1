import unittest
import numpy as np
from main import is_valid_image  # Import the is_valid_image function from your module

class TestImageValidation(unittest.TestCase):

    def test_valid_image(self):
        # Create a valid image with minimum height and width
        valid_image = np.zeros((101, 101, 3), dtype=np.uint8)
        self.assertTrue(is_valid_image(valid_image))

    def test_invalid_format(self):
        # Create an image with an invalid format (4 channels instead of 3)
        invalid_format_image = np.zeros((99, 99, 4), dtype=np.uint8)
        self.assertFalse(is_valid_image(invalid_format_image))

if __name__ == '__main__':
    unittest.main()