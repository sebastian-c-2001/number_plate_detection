import ttkbootstrap as ttk
from tkinter import filedialog
from tkinter.messagebox import showerror, askyesno
import numpy as np
import math
import cv2
import tkinter as tk
from PIL import Image, ImageTk

file_path = ""
image_label = None  # Label pentru afișarea imaginii


def open_image():
    global file_path, image_label,image
    file_path = filedialog.askopenfilename(title="Open Image File",
                                           filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])
    if file_path:
        try:
            # Deschiderea imaginii
            image = Image.open(file_path)

            # Redimensionarea imaginii pentru a se potrivi în fereastră
            max_width, max_height = 400, 300
            image.thumbnail((max_width, max_height))

            # Conversia imaginii pentru Tkinter
            tk_image = ImageTk.PhotoImage(image)

            # Dacă imaginea a fost deja afișată anterior, o eliminăm
            if image_label:
                image_label.config(image=tk_image)
                image_label.image = tk_image
            else:
                image_label = tk.Label(image_frame, image=tk_image)
                image_label.image = tk_image
                image_label.pack(padx=10, pady=10)

        except Exception as e:
            showerror("Eroare", f"Nu am putut deschide imaginea: {e}")

def detecție_numar():
    global file_path
    image_path = file_path
    if file_path:
        image=cv2.imread(image_path)








# Crearea interfeței grafice
root = tk.Tk()
root.title("Number plate detector")
root.geometry("800x400")

# Cadru pentru butoane în partea stângă
button_frame = tk.Frame(root)
button_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ns")  # poziționare în partea stângă

# Cadru pentru afișarea imaginii în partea dreaptă
image_frame = tk.Frame(root, width=400, height=300)
image_frame.grid(row=0, column=1, padx=10, pady=10)

# Buton pentru deschiderea imaginii
open_button = tk.Button(button_frame, text="Încarcă Imaginea", command=open_image)
open_button.pack(fill="x", padx=10, pady=10)

# Buton pentru detecția numărului de înmatriculare
open_button = tk.Button(button_frame, text="Detecție", command=detecție_numar())
open_button.pack(fill="x", padx=10, pady=10)


root.mainloop()
