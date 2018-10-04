import pandas
import numpy
from PIL import Image
from tqdm import tqdm



def image_to_vector(image_path, width, height):
    img = Image.open(image_path).resize((width,height)).convert("L")
    return numpy.reshape(numpy.array(img), -1)

def images_to_vectors(image_list, num_pixels):
    num_images = len(image_list)
    image_vectors = numpy.zeros((num_pixels*num_pixels, num_images))

    to_delete = []
    for column in tqdm(range(num_images), desc="Image to vector "):
        try:
            image_vectors[:,column] = image_to_vector(image_list[column], num_pixels, num_pixels)
        except:
            print("The following image could not be converted : "+image_list[column])
            to_delete.append(column)

    image_vectors = numpy.delete(image_vectors, to_delete, axis=1)

    
    return image_vectors, to_delete

def write_image_vectors_to_file(image_list, num_pixels, labels, output_file):
    header = ["Label"] + ["X"+str(i) for i in range(num_pixels*num_pixels)]
    
    image_vectors, to_delete = images_to_vectors(image_list, num_pixels)
    
    examples = numpy.append([numpy.delete(labels, to_delete)], image_vectors, axis=0)
    numpy.save(output_file, examples)
    # print("Examples saved : "+str(examples.shape))