import os

import data_processing as DP

directory = os.path.dirname(os.path.realpath(__file__))

image_path = directory+"/YOLO/abra_original.png"
out_image_path = directory + "/YOLO/abra_smaller.png"
vector_path = directory+"/YOLO/abra_vector.pkl"
# DP.save_vectorized_image(image_path, out_image_path, vector_path)

image_tensor = DP.image_to_tensor(image_path, 64, 64)
print("Tensor : "+str(image_tensor.shape))
# new_image = DP.make_square(image_path)
# new_image.show()

train_images = [
    '/Users/emiliepicardcantin/Documents/Coursera/NeuralNetwork/Datasets/Pokemon/SquareImages/Gyarados/149_Gyarados.png',
    '/Users/emiliepicardcantin/Documents/Coursera/NeuralNetwork/Datasets/Pokemon/SquareImages/Magikarp/007_Magikarp.png',
    '/Users/emiliepicardcantin/Documents/Coursera/NeuralNetwork/Datasets/Pokemon/SquareImages/Nidorino/024_Nidorino.png'
]
directory = "/Users/emiliepicardcantin/Documents/Coursera/NeuralNetwork/Datasets/Pokemon/SquareImages"

train_labels = [129, 128, 32]
train_categories = ['Gyarados', 'Magikarp', 'Nidorino']

df = DP.get_image_tensors(train_images, train_labels, train_categories, 64, "test_tensors.npy", directory)

print(df)