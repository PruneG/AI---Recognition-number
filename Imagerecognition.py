# J'installe les bibliothèques nécessaires pour la reconnaissance d'image
from PIL import Image
import numpy as np
import os 


# Je crée une classe pour exécuter diifférentes actions sur les images (l'ouvrir, la mettre en gris et noir, binarizer, sizer...)
class ImageRecognition:
    def __init__(self):
        self.reference_database = {}

    def acquire_image(self, path):
        image = Image.open(path)
        return image
    
    def convert_to_grayscale(self, image):
        gray_image = image.convert("L")
        return gray_image
    
    def binarize(self, image, threshold=128):
        binary_image = image.point(lambda p: 1 if p > threshold else 0)
        return binary_image
    
    def normalize_size(self, image, size=(10,10)):
        resized_image = image.resize(size)
        return resized_image
    
    def convert_to_matrix(self, image):
        matrix = np.array(image)
        return matrix
    
    def store_reference(self, label, matrix):
        self.reference_database[label] = matrix

    def compare_matrices(self, matrix1, matrix2):
        return np.array_equal(matrix1, matrix2)

    def prepare_image(self, path):
        image = self.acquire_image(path)
        gray_image = self.convert_to_grayscale(image)
        binary_image = self.binarize(gray_image)
        resized_image = self.normalize_size(binary_image)
        matrix = self.convert_to_matrix(resized_image)
        return matrix
        