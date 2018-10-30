import pandas
import pickle
import numpy
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

# Data augmentation
from imgaug import augmenters as iaa
seq = iaa.Sequential([
    iaa.Crop(px=(0, 10)), # crop images from each side by 0 to 16px (randomly chosen)
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.GaussianBlur(sigma=(0, 0.5)) # blur images with a sigma of 0 to 3.0
])

def make_square(image_path, fill_color=(255,255,255,255), max_size=None):
    im = Image.open(image_path)
    im.load()
    x, y = im.size
    size = max(x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    box = (int((size - x) / 2), int((size - y) / 2))

    if im.mode == "RGBA":
        new_im.paste(im, box, mask=im.split()[3])
    else:
        new_im.paste(im, box)

    # print("\tNew image size : "+str(new_im.size))
    if max_size:
        new_im = new_im.resize((max_size, max_size))
    # print("\tNew image resized : "+str(new_im.size))
    return new_im

def data_augmentation(input_folder, output_folder, category, image_list, size, repetitions=25):
    tensors = numpy.array([image_to_tensor(input_folder+i, size, size) for i in image_list])
    tensors_for_aug = numpy.repeat(tensors,repetitions,axis=0)
    
    aug_tensors = seq.augment_images(tensors_for_aug)
    
    num_digits = len(str(len(image_list)))
        
    for i,tensor in enumerate(aug_tensors):
        img = Image.fromarray(tensor, 'RGB')
        img.save(output_folder+str(i).zfill(num_digits)+"_aug_"+category+".png", "PNG")

def save_vectorized_image(image_path, out_image_path, vector_file):
    img = Image.open(image_path)#.resize((300,300))
    img_ratio = img.size[0] / float(img.size[1])
    print("Image ratio : "+str(img_ratio))
    
    img.thumbnail((300,300), Image.ANTIALIAS)
    img.save(out_image_path)

    # img.resize((300,300)).save(out_image_path)
    # vector = numpy.reshape(numpy.array(img), -1)
    # with open(vector_file, 'wb') as f:
    #     pickle.dump(vector, f)

def delete_broken_images(image_folder):
    # image_candidates = [y for x in os.walk(image_folder) for ext in ["*.jpg", "*.png"] for y in glob(os.path.join(x[0], ext))]

    image_candidates = [image_folder+f for f in os.listdir(image_folder)]
    # print(image_candidates)
    for f in image_candidates:
        try:
            Image.open(f)
        except:
            print(f+" could not be opened as an image. It will be deleted.")
            os.remove(f)

def image_to_vector(image_path, width, height):
    img = Image.open(image_path).resize((width,height)).convert("L")
    return numpy.reshape(numpy.array(img), -1)

def image_to_tensor(image_path, width, height):
    img = Image.open(image_path).resize((width,height)).convert("RGB")
    return numpy.array(img)

def images_to_vectors(image_list, num_pixels):
    num_images = len(image_list)
    image_vectors = numpy.zeros((num_pixels*num_pixels, num_images))

    for column in tqdm(range(num_images), desc="Image to vector "):
        try:
            image_vectors[:, column] = image_to_vector(image_list[column], num_pixels, num_pixels)
        except:
            print("The following image could not be converted : "+image_list[column])
            image_vectors[:, column] = numpy.full(num_pixels*num_pixels, numpy.nan)

    return image_vectors

def get_image_dataframe(image_list, labels, categories, num_pixels, output_file, directory):
    df = pandas.DataFrame({
        "Files": [f.replace(os.path.abspath(directory), "") for f in image_list], 
        "Labels": labels,
        "Categories": categories
        })

    image_vectors = images_to_vectors(image_list, num_pixels)
    num_features = image_vectors.shape[0]
    
    df_x = pandas.DataFrame({"X"+str(i):image_vectors[i,:] for i in range(num_features)})
    df = df.join(df_x)
    df.to_pickle(output_file)
    
    return df

def get_image_tensors(image_list, labels, categories, num_pixels, output_file, directory):
    tensors = {
        "Files": [f.replace(os.path.abspath(directory), "") for f in image_list], 
        "Labels": labels,
        "Categories": categories,
        "X": numpy.array([image_to_tensor(im, num_pixels, num_pixels) for im in image_list])
        }
    with open(output_file, 'wb') as f:
        pickle.dump(tensors, f)
    
    return tensors

def display_images(images, captions, header=None, width="100%"): # to match Image syntax
    if type(width)==type(1): width = "{}px".format(width)
    html = ["<table style='width:{}'><tr>".format(width)]
    if header is not None:
        html += ["<th>{}</th>".format(h) for h in header] + ["</tr><tr>"]

    for image in images:
        html.append("<td  style='text-align:center;'><img src='{}' /></td>".format(image))
    html.append("</tr><tr>")
    for caption in captions:
        html.append("<td style='text-align:center;'>{}</td>".format(caption))
    
    html.append("</tr></table>")
    return ''.join(html)