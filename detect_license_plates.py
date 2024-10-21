import cv2
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from emnist import extract_training_samples

# 1. Preprocess the input image (convert to grayscale)
def preprocess_image(image_path):
    """
    Loads an image from the provided path and converts it to grayscale.
    """
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


# 2. Detect the license plate in the image
def detect_license_plate(gray_image):
    """
    Detects the license plate from the preprocessed grayscale image.
    Uses edge detection and contour filtering to locate the plate.
    """
    # Apply edge detection (Canny)
    edges = cv2.Canny(gray_image, 100, 200)
    plt.imshow(edges, cmap='gray'),
    plt.colorbar()
    plt.show()

    # Apply thresholding to binarize the image
    _, binary = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

    plt.imshow(binary, cmap='gray'),
    plt.colorbar()
    plt.show()

    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plates = []

    # Filter contours based on size and aspect ratio
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h
        # Check if the contour matches the aspect ratio and size of a license plate
        if 2 < aspect_ratio < 5 and w > 100:
            plate = gray_image[y:y + h, x:x + w]
            plt.imshow(plate, cmap='gray'),
            plt.colorbar()
            plt.show()
            plates.append(plate)
    return plates


# 3. Segment characters from the license plate
def segment_characters(plate):
    """
    Segments individual characters from the detected license plate.
    Uses thresholding and contour detection to isolate each character.
    """
    characters = []
    # Apply thresholding to invert the plate image (background black, characters white)
    _, binary_plate = cv2.threshold(plate, 100, 255, cv2.THRESH_BINARY_INV)

    # Detect contours for each character in the license plate
    contours, _ = cv2.findContours(binary_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []

    # Extract each character based on the bounding rectangle of its contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if 20 < h < 40:  # Ignore small contours (noise)
            char = binary_plate[y:y + h, x:x + w]
            plt.imshow(char, cmap='gray'),
            plt.colorbar()
            plt.show()
            characters.append(char)

    return characters


# 4. Classify each segmented character using a pre-trained SVM
def classify_characters(characters, classifier=None):
    """
    Classifies each character using the provided classifier.
    Characters are resized and flattened before being passed to the classifier.
    """
    recognized_text = ""

    for char in characters:
        # Resize each character to 28x28, similar to EMNIST size
        char_resized = cv2.resize(char, (28, 28))
        char_flatten = char_resized.flatten().reshape(1, -1)

        # Predict the character using the trained classifier
        # recognized_char = classifier.predict(char_flatten)
        # recognized_text += str(recognized_char[0])

    return recognized_text

# Function to train an SVM classifier on the MNIST dataset
def train_svm_on_mnist():
    """
    Trains an SVM classifier on the EMNIST dataset and returns the trained classifier.
    """
    # Load the EMNIST dataset (for both digits and letters)
    X_emnist, y_emnist = extract_training_samples('balanced')

    # Flatten the images and split into train/test sets
    X_emnist = X_emnist.reshape((len(X_emnist), -1))
    X_train, X_test, y_train, y_test = train_test_split(X_emnist, y_emnist, test_size=0.2, random_state=42)

    # Train SVM classifier on EMNIST dataset
    classifier = svm.SVC(gamma='scale')
    classifier.fit(X_train, y_train)

    # Test the classifier on test data and print accuracy
    y_pred = classifier.predict(X_test)
    print(f"Accuracy on EMNIST test set: {accuracy_score(y_test, y_pred):.2f}")

    return classifier

# Main function to run the steps
def detect_license_plates(image_path):
    # Step 1: Preprocess the input image (grayscale conversion)
    gray_image = preprocess_image(image_path)

    # Step 2: Detect the license plate in the image
    plates = detect_license_plate(gray_image)

    # Step 4: Classify each segmented character using a pre-trained SVM
    # Train the SVM classifier on the EMNIST dataset
    # classifier = train_svm_on_mnist()

    for plate in plates:
        # Step 3: Segment characters from the license plate
        characters = segment_characters(plate)

        # Now use the trained classifier for recognizing license plate characters
        recognized_text = classify_characters(characters)

        print("Recognized License Plate: ", recognized_text)
    else:
        print("License plate could not be detected.")


if __name__ == "__main__":
    # Example usage
    image_path = 'images/img_1.png'  # Replace with the actual path to the image
    detect_license_plates(image_path)
