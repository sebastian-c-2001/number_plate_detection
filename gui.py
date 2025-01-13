import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from joblib import load
import matplotlib.pyplot as plt
import imutils
from skimage.feature import hog
svm = load('svm_model.joblib')


def filter_similar_contours(contours, overlap_threshold=0.5):
    # Memorarea contururilor filtrare
    filtered_contours = []
    bounding_boxes = []

    # Step 1: Crearea de bounding boxes pentru toate contururile
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

    # Step 2: Verificarea suprapunerilor si filtrarea
    for i, box1 in enumerate(bounding_boxes):
        keep = True
        x1, y1, w1, h1 = box1
        area1 = w1 * h1

        for j, box2 in enumerate(bounding_boxes):
            if i == j:
                continue  # Verificare pentru a nu fi exact același bounding box

            x2, y2, w2, h2 = box2
            area2 = w2 * h2

            # Step 3: Calculam IoU
            inter_x1 = max(x1, x2)
            inter_y1 = max(y1, y2)
            inter_x2 = min(x1 + w1, x2 + w2)
            inter_y2 = min(y1 + h1, y2 + h2)

            # Daca nu avem suprapuneri continuam
            if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
                continue

            # Calculam Aria de Intersectie si Aria de Uniune
            intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            union_area = area1 + area2 - intersection_area

            # Intersection over Union (IoU)
            iou = intersection_area / union_area

            # Step 4: Dacă se suprapune mai mult decât pragul ales
            if iou > overlap_threshold:
                # Dacă se suprapune păstrăm conturul mai mare
                if area1 < area2:
                    keep = False
                    break

        if keep:
            filtered_contours.append(contours[i])

    return filtered_contours
def clear_output_text():
    output_text.delete(1.0, tk.END)

def make_feature(img):
    return hog(
        img, orientations=12,
        pixels_per_cell=(14, 14),
        cells_per_block=(1, 1),
        block_norm="L2"
    )
# Funcție pentru încărcarea imaginii
def load_image():
    clear_output_text()
    plt.close('all')
    global file_path
    file_path = filedialog.askopenfilename(title="Open Image File",
                                           filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        image = Image.open(file_path)
        image = image.resize((320, 240))  # Redimensionare pentru afișare
        photo = ImageTk.PhotoImage(image)
        label_image.config(image=photo)
        label_image.image = photo

def load_dataset(directory):
    class_dict = {}
    data = []
    for i, dir in enumerate(sorted(os.listdir(directory))):
        class_dict[i] = dir
        for impath in sorted(os.listdir(os.path.join(directory, dir))):
            data_item = {
                "path": os.path.join(directory,dir,impath),
                "label": i
            }
            data.append(data_item)
    return data, class_dict
def detectie_numar():
    global file_path,dsize,svm
    data_train, class_dict = load_dataset('data/training_data')
    image = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    plt.figure(), plt.imshow(image), plt.show(), plt.show()
    plt.figure(), plt.imshow(gray_image, cmap='gray'), plt.show()


    edged = cv2.Canny(gray_image, 30, 200)  # Edge detection
    plt.figure(), plt.imshow(edged, cmap='gray'), plt.show(), plt.show()

    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    boxes = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, peri * 0.05, True)
        if len(approx) == 4:  # Check if it has 4 sides
            x, y, w, h = cv2.boundingRect(approx)
            # Calculare raport lungime/latime
            aspect_ratio = w / float(h)
            if 2.0 < aspect_ratio < 5.0:  # Raport tipic
                boxes.append(approx)

    final_boxes = filter_similar_contours(boxes, overlap_threshold=0.5)
    validator=0
    if len(final_boxes) > 0:
        for box in final_boxes:
            mask = np.zeros(gray_image.shape, np.uint8)
            cv2.drawContours(mask, [box], 0, (255, 255, 255), -1)
            new_image = cv2.bitwise_and(gray_image, gray_image, mask=mask)
            # plt.figure(), plt.imshow(mask, cmap='gray'), plt.title("Mask"), plt.show()
            # plt.figure(), plt.imshow(new_image, cmap='gray'), plt.title("New Img"), plt.show()

            (x, y) = np.where(mask == 255)
            (x1, y1) = (np.min(x), np.min(y))
            (x2, y2) = (np.max(x), np.max(y))
            cropped_image = gray_image[x1:x2 + 1, y1:y2 + 1]
            cropped_image = cv2.resize(cropped_image, (300, 80))
            plt.figure(), plt.imshow(cropped_image, cmap='gray'), plt.title("Detected"), plt.show()

            blurred_plate = cv2.GaussianBlur(cropped_image, (3, 3), 0)
            edges_plate = cv2.Canny(blurred_plate, 50, 150)
            plt.figure(), plt.imshow(blurred_plate, cmap='gray'), plt.title("Blurred_plate"), plt.show()
            plt.figure(), plt.imshow(edges_plate, cmap='gray'), plt.title("Canny"),plt.show()

            # kernel = np.ones((3, 3), np.uint8)
            # dilated = cv2.dilate(edges_plate, kernel, iterations=1)
            # eroded = cv2.erode(dilated, kernel, iterations=1)
            # plt.figure(), plt.imshow(eroded, cmap='gray'), plt.show()

            contours, _ = cv2.findContours(edges_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Sort contours from left to right based on bounding box positions
            bounding_boxes = [cv2.boundingRect(c) for c in contours]
            sorted_boxes = sorted(bounding_boxes, key=lambda x: x[0])

            characters = []
            image_with_boxes = cropped_image.copy()
            nr=0;
            # Iterăm prin sorted_boxes și desenăm fiecare bounding box
            for i in sorted_boxes:
                x, y, w, h = i
                # Desenează un dreptunghi (bounding box)
                if  h > 30 and 5 < w < 50:
                    cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)  # verde, grosime 2

                    nr+=1
            print("Nr de caractere = ", nr)

            # Afișare imagine cu bounding box-uri
            plt.figure(), plt.imshow(image_with_boxes, cmap='gray'), plt.title("Bounding Boxes"), plt.show()
            for i, (x, y, w, h) in enumerate(sorted_boxes):
                # Filter by reasonable character dimensions
                if h > 40 and 5 < w < 50:
                    char = cropped_image[y:y + h, x:x + w]
                    im, thre = cv2.threshold(char, 100, 255,
                                             cv2.THRESH_BINARY_INV)
                    thre = np.pad(thre, (2, 2), 'constant', constant_values=(0, 0))
                    res = cv2.resize(thre, dsize=dsize, interpolation=cv2.INTER_CUBIC)  # resize 30, 80

                    #plt.figure(), plt.imshow(res, cmap='gray'), plt.title("Char Img"), plt.show()
                    characters.append(res)

            if(len(characters)<=1):
                characters = []
                for i, (x, y, w, h) in enumerate(sorted_boxes):
                    # Filter by reasonable character dimensions
                    if h > 30 and 5 < w < 50:
                        char = cropped_image[y:y + h, x:x + w]
                        im, thre = cv2.threshold(char, 100, 255,
                                                 cv2.THRESH_BINARY_INV)
                        thre = np.pad(thre, (2, 2), 'constant', constant_values=(0, 0))
                        res = cv2.resize(thre, dsize=dsize, interpolation=cv2.INTER_CUBIC)  # resize 30, 80

                        plt.figure(), plt.imshow(res, cmap='gray'), plt.title("Char Img"), plt.show()
                        characters.append(res)
            # (it s a problem in license plates that have a sticker between numbers -> doesn't take the second one because it s too wide with sticker next)
            if len(characters) > 5:
                text = ""
                X_feature = np.array([make_feature(i) for i in characters], np.float32)

                # Predict the character using the trained classifier
                recognized_text = svm.predict(X_feature)
                for i in recognized_text:
                    text = text + class_dict[i]
                text_nou="Numarul de inmatriculare detectat: " + text
                validator=1;
                output_text.insert(tk.END, text_nou)
                font = cv2.FONT_HERSHEY_SIMPLEX
                rez = image.copy()
                print("Bounding box ul este:",box)

                # cv2.putText(rez, text=text, org=(box[0][0][0], box[2][0][1] + 70), fontFace=font, fontScale=2,
                #             color=(0, 255, 0), thickness=3, lineType=cv2.LINE_AA)
                # if (abs(box[0][0][1] - box [2][0][1])<50):
                #     cv2.rectangle(rez, tuple(box[0][0]), tuple(box[3][0]), (0, 255, 0), 3)
                # else:
                #     cv2.rectangle(rez, tuple(box[0][0]), tuple(box[2][0]), (0, 255, 0), 3)
                x_coords = [point[0][0] for point in box]  # Coordonatele X ale tuturor celor 4 puncte
                y_coords = [point[0][1] for point in box]  # Coordonatele Y ale tuturor celor 4 puncte

                # Calculăm colțul stânga sus (minim x și y) și colțul dreapta jos (maxim x și y)
                x1, y1 = min(x_coords), min(y_coords)  # Punctul minim
                x2, y2 = max(x_coords), max(y_coords)  # Punctul maxim

                # Desenăm dreptunghiul pe imagine folosind coordonatele calculate
                cv2.rectangle(rez, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Culoare verde, grosime 3

                plt.figure(), plt.imshow(rez), plt.show()
                rez = Image.fromarray(rez)
                img = rez.resize((320, 240))  # Redimensionare pentru afișare
                photo = ImageTk.PhotoImage(img)
                label_image.config(image=photo)
                label_image.image = photo
                plt.figure(), plt.imshow(rez), plt.show()
    #         else:
    #             output_text.insert(tk.END, "Nu se poate citi placuta")
    # else:
    #     output_text.insert(tk.END, "Nu se poate detecta placuta")
    if validator==0:
        output_text.insert(tk.END, "Nu se poate detecta placuta")

# Variabila globală pentru calea imaginii
image_path = None
dsize = (30, 80)

# Crearea ferestrei principale
root = tk.Tk()
root.title("Number plate detection")
root.geometry("600x400")

# Etichetă pentru afișarea imaginii încărcate
label_image = tk.Label(root, text="Imaginea va apărea aici")
label_image.grid(row=0, column=1, padx=10, pady=10)

# Frame pentru butoane
button_frame = tk.Frame(root)
button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")

# Buton pentru încărcarea imaginii
load_button = tk.Button(button_frame, text="Încarcă Imaginea", command=load_image)
load_button.pack(fill="x", padx=10, pady=10)

# Buton pentru aplicarea SVM
apply_button = tk.Button(button_frame, text="Detecteaza placuta", command=detectie_numar)
apply_button.pack(fill="x", padx=10, pady=10)

# Text box pentru afișarea rezultatului
output_text = tk.Text(root, height=5, width=60)
output_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

# Rularea buclei principale a aplicației
root.mainloop()



