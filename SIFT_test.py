import joblib

file = joblib.load("./Cat_dog/SIFT/data/dataset/raw_description.joblib")

print(file[0][0])