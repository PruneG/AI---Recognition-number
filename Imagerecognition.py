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
        
    # Je crée une fonction qui construit une base de données de référence à partir d'un dossier d'images et du mapping des labels
    def build_reference_database(self, folder_path, label_mapping):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if os.path.isfile(file_path)and file_name.lower().endswith((".png", ".jpg", ".jpeg")):
                matrix = self.prepare_image(file_path)
                file_label = os.path.splitext(file_name)[0]
                
                if file_label in label_mapping:
                    real_label = label_mapping[file_label]
                    self.store_reference(real_label, matrix)

    # Je crée une fonction qui calcule la distance entre deux matrices pour évaluer la similarité entre l'image test et les références
    def calculate_distance(self, matrix1, matrix2):
        distance = np.sum(np.abs(matrix1 - matrix2))
        return distance
    
    # Je crée une fonction qui reconnaît une image en comparant sa matrice avec celles de la base de données de référence et en retournant le label de la référence la plus proche et la distance associée
    def recognize_image(self, path):
        test_matrix = self.prepare_image(path)
        best_label = None
        best_distance = float("inf")

        for label, ref_matrix in self.reference_database.items():
            distance = self.calculate_distance(test_matrix, ref_matrix)

            if distance < best_distance:
                best_distance = distance
                best_label = label
                
        return best_label, best_distance