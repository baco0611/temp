import tkinter as tk
from tkinter import filedialog
from tkinterdnd2 import TkinterDnD, DND_FILES
from PIL import Image, ImageTk
import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from keras.preprocessing import image
from app_config import *

def convert_labe_to_text(i):
    if i == 0:
        return "CAT"
    else:
        return "DOG"
    return i

def load_image(file_path=None):
    if not file_path:
        file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions and determine the resize dimensions
        img_height, img_width, _ = img_rgb.shape
        if img_width > img_height:
            new_width = 200
            new_height = int(new_width * img_height / img_width)
        else:
            new_height = 200
            new_width = int(new_height * img_width / img_height)
        
        img_resized = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img_resized = Image.fromarray(img_resized)
        img_tk = ImageTk.PhotoImage(img_resized)
        panel.img_tk = img_tk
        panel.config(image=img_tk)
        
        # SIFT classification
        sift_result = classify_image_sift(img)
        sift_label.config(text=f"SIFT: {convert_labe_to_text(sift_result)}")
        
        # VGG8 classification
        vgg_result = classify_image_vgg(img)
        vgg_label.config(text=f"VGG8: {convert_labe_to_text(vgg_result)}")

        #VGG8_SVM classification
        dims = 4096
        vector_4096_dims = classify_extracted_feature(img, dims)
        vector_4096_label.config(text=f"VGG8_4096_dims: {convert_labe_to_text(vector_4096_dims)}")
        dims = 1024
        vector_1024_dims = classify_extracted_feature(img, dims)
        vector_1024_label.config(text=f"VGG8_1024_dims: {convert_labe_to_text(vector_1024_dims)}")


        pca_4096 = []
        for i in range(200, 600, 100):
            pca_4096.append(classify_pca_extract(img, 4096, i))
        
        vector_4096_200_label.config(text=f"4096_200_dims: {convert_labe_to_text(vector_4096_dims)}")
        vector_4096_300_label.config(text=f"4096_300_dims: {convert_labe_to_text(vector_4096_dims)}")
        vector_4096_400_label.config(text=f"4096_400_dims: {convert_labe_to_text(vector_4096_dims)}")
        vector_4096_500_label.config(text=f"4096_500_dims: {convert_labe_to_text(vector_4096_dims)}")
        vector_4096_200_label.config(text=f"4096_200_dims: {convert_labe_to_text(pca_4096[0])}")
        vector_4096_300_label.config(text=f"4096_300_dims: {convert_labe_to_text(pca_4096[1])}")
        vector_4096_400_label.config(text=f"4096_400_dims: {convert_labe_to_text(pca_4096[2])}")
        vector_4096_500_label.config(text=f"4096_500_dims: {convert_labe_to_text(pca_4096[3])}")

        pca_1024 = []
        for i in range(200, 600, 100):
            pca_1024.append(classify_pca_extract(img, 1024, i))
    

        vector_1024_200_label.config(text=f"1024_200_dims: {convert_labe_to_text(vector_1024_dims)}")
        vector_1024_300_label.config(text=f"1024_300_dims: {convert_labe_to_text(vector_1024_dims)}")
        vector_1024_400_label.config(text=f"1024_400_dims: {convert_labe_to_text(vector_1024_dims)}")
        vector_1024_500_label.config(text=f"1024_500_dims: {convert_labe_to_text(vector_1024_dims)}")
        vector_1024_200_label.config(text=f"1024_200_dims: {convert_labe_to_text(pca_1024[0])}")
        vector_1024_300_label.config(text=f"1024_300_dims: {convert_labe_to_text(pca_1024[1])}")
        vector_1024_400_label.config(text=f"1024_400_dims: {convert_labe_to_text(pca_1024[2])}")
        vector_1024_500_label.config(text=f"1024_500_dims: {convert_labe_to_text(pca_1024[3])}")

def drop(event):
    files = event.data.split()
    for file in files:
        load_image(file)

app = TkinterDnD.Tk()
app.title("CAT_DOG CLASSIFICTION")
app.geometry("500x800")
app.configure(bg="#12377F")

# Title
title = tk.Label(app, text="CAT_DOG CLASSIFICTION", font=("Helvetica", 24), bg="#12377F", fg="white")
title.pack(fill=tk.X)

# Image panel
panel = tk.Label(app, text="import picture", bg="white")
panel.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Import button
import_button = tk.Button(app, text="Import Image", command=lambda: load_image(), font=("Helvetica", 14), bg="#12377F", fg="white")
import_button.pack(pady=10)

# SIFT and VGG labels
sift_label = tk.Label(app, text="SIFT: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
sift_label.pack()
vgg_label = tk.Label(app, text="VGG8: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vgg_label.pack()

vector_4096_label = tk.Label(app, text="VGG8_4096_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_4096_label.pack()
vector_1024_label = tk.Label(app, text="VGG8_1024_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_1024_label.pack()

vector_4096_200_label = tk.Label(app, text="4096_200_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_4096_200_label.pack()
vector_4096_300_label = tk.Label(app, text="4096_300_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_4096_300_label.pack()
vector_4096_400_label = tk.Label(app, text="4096_400_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_4096_400_label.pack()
vector_4096_500_label = tk.Label(app, text="4096_500_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_4096_500_label.pack()

vector_1024_200_label = tk.Label(app, text="1024_200_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_1024_200_label.pack()
vector_1024_300_label = tk.Label(app, text="1024_300_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_1024_300_label.pack()
vector_1024_400_label = tk.Label(app, text="1024_400_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_1024_400_label.pack()
vector_1024_500_label = tk.Label(app, text="1024_500_dims: ........", font=("Helvetica", 20), bg="#12377F", fg="white")
vector_1024_500_label.pack()

# Enable drop target
app.drop_target_register(DND_FILES)
app.dnd_bind('<<Drop>>', drop)

app.mainloop()
