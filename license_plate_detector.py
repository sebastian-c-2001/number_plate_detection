import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from joblib import load, dump
import matplotlib.pyplot as plt
import imutils
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score


class LicensePlateDetector:
    def __init__(self, train_if_needed=True):
        # Initialize constants
        self.CHAR_SIZE = (30, 80)
        self.BLUR_SIZE = (3, 3)
        self.THRESHOLD_VAL = 100
        self.RANDOM_SEED = 112

        # Try to load or train the model
        try:
            self.svm = load('svm_model.joblib')
        except FileNotFoundError:
            if train_if_needed:
                print("Model not found. Training new model...")
                self.train_model()
            else:
                raise FileNotFoundError("SVM model not found and training disabled")

        # Initialize GUI components
        self.root = tk.Tk()
        self.root.title("License Plate Detection")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        self.root.wm_positionfrom("program")

        # Initialize state
        self.current_image_path = None

        # Setup GUI
        self._setup_gui()

    def train_model(self):
        """Train the SVM model for character recognition"""
        # Load training and testing data
        data_train, class_dict = self._load_dataset('data/training_data')
        data_test, _ = self._load_dataset('data/testing_data')

        # Preprocess training and testing data
        x_train, y_train = self._preprocess_dataset(data_train)
        x_test, y_test = self._preprocess_dataset(data_test)

        # Extract HOG features
        x_train_feature = np.array([self._make_feature(i) for i in x_train], np.float32)
        x_test_feature = np.array([self._make_feature(i) for i in x_test], np.float32)

        # Train SVM
        self.svm = svm.LinearSVC(C=0.5, random_state=self.RANDOM_SEED)
        self.svm.fit(x_train_feature, y_train)

        # Evaluate model
        y_pred = self.svm.predict(x_test_feature)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"SVM Model Accuracy: {accuracy:.4f}")

        # Save model
        dump(self.svm, 'svm_model.joblib')
        print("Model saved successfully.")

    @staticmethod
    def _preprocess_image(img):
        """Preprocess a single image for training or prediction"""
        blurred = cv2.GaussianBlur(img, (3, 3), 0)  # smoothing
        _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)  # binarization
        thresh = np.pad(thresh, (2, 2), 'constant', constant_values=(0, 0))
        resized = cv2.resize(thresh, dsize=(30, 80), interpolation=cv2.INTER_CUBIC)
        return resized

    def _preprocess_dataset(self, data):
        """Preprocess entire dataset for training"""
        processed_images = []
        labels = []
        for item in data:
            img = cv2.imread(item['path'], cv2.IMREAD_GRAYSCALE)
            processed = self._preprocess_image(img)
            processed_images.append(processed)
            labels.append(item['label'])
        return np.stack(processed_images), np.array(labels)

    def _setup_gui(self):
        """Setup all GUI components"""
        # Create image display label
        self.label_image = tk.Label(self.root, text="Image will appear here")
        self.label_image.grid(row=0, column=1, padx=10, pady=10)

        # Create button frame
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

        # Create buttons
        load_button = tk.Button(button_frame, text="Load Image", command=self._load_image)
        load_button.pack(fill="x", padx=10, pady=10)

        detect_button = tk.Button(button_frame, text="Detect License Plate",
                                  command=self._detect_license_plate)
        detect_button.pack(fill="x", padx=10, pady=10)

        train_button = tk.Button(button_frame, text="Retrain Model",
                                 command=self._retrain_model)
        train_button.pack(fill="x", padx=10, pady=10)

        # Create output text box
        self.output_text = tk.Text(self.root, height=5, width=60, state='disabled')
        self.output_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

    def _retrain_model(self):
        """Handler for retrain button"""
        if messagebox.askyesno("Retrain Model",
                               "Are you sure you want to retrain the model? This may take a while."):
            try:
                self.train_model()
                messagebox.showinfo("Success", "Model retrained successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to retrain model: {str(e)}")

    # When you need to update the text:
    def _update_output_text(self, message):
        self.output_text.configure(state='normal')  # Temporarily enable to update
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, message)
        self.output_text.configure(state='disabled')  # Disable again

    @staticmethod
    def _make_feature(img):
        """
        Extract HOG features from image
        Args:
            img: Input image
        Returns:
            HOG features
        """
        return hog(
            img, orientations=12,
            pixels_per_cell=(14, 14),
            cells_per_block=(1, 1),
            block_norm="L2"
        )

    @staticmethod
    def _filter_similar_contours(contours, overlap_threshold=0.5):
        """
        Filter out overlapping contours based on IoU (Intersection over Union)
        Args:
            contours: List of contours to filter
            overlap_threshold: IoU threshold for filtering (default: 0.5)
        Returns:
            List of filtered contours
        """
        filtered_contours = []
        bounding_boxes = []

        # Create bounding boxes for all contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxes.append((x, y, w, h))

        # Check for overlaps and filter
        for i, box1 in enumerate(bounding_boxes):
            keep = True
            x1, y1, w1, h1 = box1
            area1 = w1 * h1

            for j, box2 in enumerate(bounding_boxes):
                if i == j:
                    continue  # Skip comparing with itself

                x2, y2, w2, h2 = box2
                area2 = w2 * h2

                # Calculate IoU
                inter_x1 = max(x1, x2)
                inter_y1 = max(y1, y2)
                inter_x2 = min(x1 + w1, x2 + w2)
                inter_y2 = min(y1 + h1, y2 + h2)

                # Skip if no overlap
                if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                    continue

                # Calculate Intersection and Union areas
                intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
                union_area = area1 + area2 - intersection_area

                # Calculate Intersection over Union (IoU)
                iou = intersection_area / union_area

                # If overlap is greater than threshold
                if iou > overlap_threshold:
                    # Keep the larger contour
                    if area1 < area2:
                        keep = False
                        break

            if keep:
                filtered_contours.append(contours[i])

        return filtered_contours

    @staticmethod
    def _load_dataset(directory):
        """
        Load training dataset from directory
        Args:
            directory: Path to dataset directory
        Returns:
            Tuple of (data list, class dictionary)
        """
        class_dict = {}
        data = []
        for i, dir in enumerate(sorted(os.listdir(directory))):
            class_dict[i] = dir
            for impath in sorted(os.listdir(os.path.join(directory, dir))):
                data_item = {
                    "path": os.path.join(directory, dir, impath),
                    "label": i
                }
                data.append(data_item)
        return data, class_dict

    def _load_image(self):
        """Load and display image in the GUI"""
        self._update_output_text('')
        # plt.close('all')

        self.current_image_path = filedialog.askopenfilename(
            title="Open Image File",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )

        if self.current_image_path:
            image = Image.open(self.current_image_path)
            image = image.resize((320, 240))  # Resize for display
            photo = ImageTk.PhotoImage(image)
            self.label_image.config(image=photo)
            self.label_image.image = photo  # Keep a reference!

    def _detect_license_plate(self):
        """Main function for license plate detection and recognition"""
        self._update_output_text('')

        if not self.current_image_path:
            self._update_output_text('You need to load the image first.')
            return

        data_train, class_dict = self._load_dataset('data/training_data')
        image = cv2.cvtColor(cv2.imread(self.current_image_path, cv2.IMREAD_COLOR),
                             cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Display original and grayscale images
        # plt.figure(), plt.imshow(image), plt.show()
        # plt.figure(), plt.imshow(gray_image, cmap='gray'), plt.show()

        # Edge detection
        edged = cv2.Canny(gray_image, 30, 200)
        # plt.figure(), plt.imshow(edged, cmap='gray'), plt.show()

        # Find contours
        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

        # Find potential license plate rectangles
        boxes = []
        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, peri * 0.05, True)
            if len(approx) == 4:  # Check for rectangular shape
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                if 2.0 < aspect_ratio < 5.0:  # Typical license plate aspect ratio
                    boxes.append(approx)

        final_boxes = self._filter_similar_contours(boxes, overlap_threshold=0.5)
        detection_successful = False

        if len(final_boxes) > 0:
            for box in final_boxes:
                # Create mask and extract plate region
                mask = np.zeros(gray_image.shape, np.uint8)
                cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
                # plate_region = cv2.bitwise_and(gray_image, gray_image, mask=mask)
                # plt.figure(), plt.imshow(plate_region, cmap='gray'), plt.title("Detected"), plt.show()

                # Crop and process plate region
                (x, y) = np.where(mask == 255)
                (x1, y1) = (np.min(x), np.min(y))
                (x2, y2) = (np.max(x), np.max(y))
                cropped_plate = gray_image[x1:x2 + 1, y1:y2 + 1]
                cropped_plate = cv2.resize(cropped_plate, (300, 80))
                # plt.figure(), plt.imshow(cropped_plate, cmap='gray'), plt.title("Detected"), plt.show()

                # Apply preprocessing
                blurred_plate = cv2.GaussianBlur(cropped_plate, (3, 3), 0)
                edges_plate = cv2.Canny(blurred_plate, 50, 150)
                # plt.figure(), plt.imshow(blurred_plate, cmap='gray'), plt.title("Blurred_plate"), plt.show()
                # plt.figure(), plt.imshow(edges_plate, cmap='gray'), plt.title("Canny"), plt.show()

                # Find character contours
                contours, _ = cv2.findContours(edges_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                bounding_boxes = [cv2.boundingRect(c) for c in contours]
                sorted_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # Sort left to right

                # Extract and process individual characters
                characters = self._extract_characters(cropped_plate, sorted_boxes)

                # Recognize characters if enough are found
                if len(characters) > 5:
                    # Extract features and predict
                    X_feature = np.array([self._make_feature(i) for i in characters], np.float32)
                    recognized_text = self.svm.predict(X_feature)

                    # Combine predictions into final text
                    text = "".join(class_dict[i] for i in recognized_text)
                    output_message = "Detected License Plate: " + text
                    detection_successful = True
                    self._update_output_text(output_message)

                    # Draw result on image
                    result_image = self._draw_result(image, box)

                    # Update display
                    # plt.figure(), plt.imshow(result_image), plt.show()
                    result_pil = Image.fromarray(result_image)
                    result_pil = result_pil.resize((320, 240))
                    photo = ImageTk.PhotoImage(result_pil)
                    self.label_image.config(image=photo)
                    self.label_image.image = photo

        if not detection_successful:
            self._update_output_text("License plate detection failed")

    def _extract_characters(self, plate_image, sorted_boxes):
        """Extract individual characters from the plate image"""
        characters = []

        # First attempt with stricter height threshold
        for x, y, w, h in sorted_boxes:
            if h > 40 and 5 < w < 50:
                char = plate_image[y:y + h, x:x + w]
                _, thresh = cv2.threshold(char, 100, 255, cv2.THRESH_BINARY_INV)
                thresh = np.pad(thresh, (2, 2), 'constant', constant_values=(0, 0))
                resized = cv2.resize(thresh, dsize=self.CHAR_SIZE, interpolation=cv2.INTER_CUBIC)
                characters.append(resized)

        # Second attempt with relaxed height threshold if needed
        if len(characters) <= 1:
            characters = []
            for x, y, w, h in sorted_boxes:
                if h > 30 and 5 < w < 50:
                    char = plate_image[y:y + h, x:x + w]
                    _, thresh = cv2.threshold(char, 100, 255, cv2.THRESH_BINARY_INV)
                    thresh = np.pad(thresh, (2, 2), 'constant', constant_values=(0, 0))
                    resized = cv2.resize(thresh, dsize=self.CHAR_SIZE, interpolation=cv2.INTER_CUBIC)
                    # plt.figure(), plt.imshow(resized, cmap='gray'), plt.title("Character"), plt.show()
                    characters.append(resized)

        return characters

    @staticmethod
    def _draw_result(image, box):
        """Draw the detected license plate on the image"""
        result_image = image.copy()
        x_coords = [point[0][0] for point in box]
        y_coords = [point[0][1] for point in box]
        x1, y1 = min(x_coords), min(y_coords)
        x2, y2 = max(x_coords), max(y_coords)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return result_image

    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """Main entry point for the application"""
    try:
        app = LicensePlateDetector(train_if_needed=True)
        app.run()
    except Exception as e:
        print(f"Error starting application: {str(e)}")


if __name__ == "__main__":
    main()