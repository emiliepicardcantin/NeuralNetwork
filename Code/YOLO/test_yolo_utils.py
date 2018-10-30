import unittest
import os
from PIL import Image
import scipy.misc
from matplotlib.pyplot import imshow

import yolo_utils as YU

directory = os.path.dirname(os.path.realpath(__file__))

class TestDrawBoxes(unittest.TestCase):
    def test_draw_simple_boxes(self):
        image_file = "pokemon_test.jpg"
        out_file = directory+"/out_"+image_file
        
        img = Image.open(directory+"/"+image_file)
        out_scores = [0.7]
        out_boxes = [[120, 90, 130, 134]]
        out_classes = [1]
        class_names = {1:"Pikachu"}
        colors = {1: (255, 0, 0, 255)}
        
        YU.draw_boxes(img, out_scores, out_boxes, out_classes, class_names, colors)

        img.save(out_file, quality=90)
        output_image = scipy.misc.imread(out_file)
        imshow(output_image)


if __name__ == '__main__':
    unittest.main()