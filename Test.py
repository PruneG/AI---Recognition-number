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


# Maintenant que j'ai crée une fonction qui construit une base de données de référence à partir d'un dossier d'images et du mapping des labels
# Je vais tester la reconnaissance d'une image en utilisant cette base de données
label_mapping = {
    "Group 12": "1",
    "Group 13": "2",
    "Group 14": "3",
    "Group 15": "4",
    "Group 16": "5",
    "Group 17": "6",
    "Group 18": "7",
    "Group 19": "8",
    "Group 20": "9"
}

recognizer.build_reference_database("Images", label_mapping)

print("Références enregistrées :")
print(recognizer.reference_database.keys())

label, distance = recognizer.recognize_image("Images/Group test.png")

print("Chiffre reconnu :", label)
print("Distance :", distance)