from Imagerecognition import ImageRecognition

recognizer = ImageRecognition()


# J'acquiers une image, je la convertis en niveaux de gris, puis en binaire, je normalise sa taille et je la convertis en matrice

image = recognizer.acquire_image("Images/Group 12.png")
gray_image = recognizer.convert_to_grayscale(image)
binary_image = recognizer.binarize(gray_image)
resized_image = recognizer.normalize_size(binary_image)
matrix = recognizer.convert_to_matrix(resized_image)
recognizer.store_reference("1", matrix)

print("Matrice de l'image de référence :")
print(matrix)

# Je teste la comparaison entre la matrice de l'image de référence et celle d'une nouvelle image
test_image = recognizer.acquire_image("Images/Group 12.png")
test_gray_image = recognizer.convert_to_grayscale(test_image)
test_binary_image = recognizer.binarize(test_gray_image)
test_resized_image = recognizer.normalize_size(test_binary_image)
test_matrix = recognizer.convert_to_matrix(test_resized_image)

# Comparaison avec l'image de référence
result = recognizer.compare_matrices(recognizer.reference_database["1"], test_matrix)

print("Résultat de la comparaison avec une image différente :", result)