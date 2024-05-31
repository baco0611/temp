import os 
import cv2
from app_config import *
import pandas as pd

folders = ["raw", "negative", "resize", "rotate", "flip"]
folders = ["raw"]

data_full = []

for folder in folders:
    for _, dirs, _ in os.walk(f"./dataset/{folder}"):
        for directory in dirs:
            folder_dirs = f"./dataset/{folder}/{directory}"

            for _, _, files in os.walk(folder_dirs):
                for file in files:
                    file_path = os.path.join(folder_dirs, file)

                    data = [folder, directory, file]

                    img = cv2.imread(file_path)

                    sift_result = classify_image_sift(img)
                    data.append(sift_result)

                    vgg_result = classify_image_vgg(img)
                    data.append(vgg_result)

                    #VGG8_SVM classification
                    dims = 4096
                    vector_4096_dims = classify_extracted_feature(img, dims)
                    data.append(vector_4096_dims)
                    dims = 1024
                    vector_1024_dims = classify_extracted_feature(img, dims)
                    data.append(vector_1024_dims)


                    pca_4096 = []
                    for i in range(200, 600, 100):
                        data.append(classify_pca_extract(img, 4096, i))
                    

                    pca_1024 = []
                    for i in range(200, 600, 100):
                        data.append(classify_pca_extract(img, 1024, i))

                    data_full.append(data)



# Đặt tên cho các cột
columns = ["category", "label", "file_path", "sift", "vgg8", "vgg8_4096", "vgg8_1024", 
           "4096_200", "4096_300", "4096_400", "4096_500", "1024_200", "1024_300", "1024_400", "1024_500"]

# Tạo DataFrame và lưu vào tệp CSV
df = pd.DataFrame(data_full, columns=columns)
df.to_csv("output.csv", index=False)