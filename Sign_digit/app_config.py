import cv2
import joblib
from keras.models import load_model, Model
import numpy as np

# Load models and codebook
sift = cv2.SIFT_create()
codebook = joblib.load('./SIFT/data/model/20240520_process_200_codebook.joblib')  
SIFT_model = joblib.load('./SIFT/data/model/SVM_20240520_process_200_model.joblib') 
vgg_model = load_model("./VGG8/VGG8/model/20240520_1_VGG8_process_30_CNN_model.h5")

def extract_sift_features(img):
    kp, des = sift.detectAndCompute(img, None)
    return des

def represent_image_features(image_descriptors, codebook):
# image_descriptors: a list of local features

    from scipy.cluster.vq import vq

    # Map each descriptor to the nearest codebook entry
    img_visual_words, distance = vq(image_descriptors, codebook)
    
    # create a frequency vector for each image
    k = codebook.shape[0]
    image_frequency_vector = np.zeros(k)
    for word in img_visual_words:
        image_frequency_vector[word] += 1
    return image_frequency_vector

def classify_image_sift(img):
    des = extract_sift_features(img)
    feature_vector = [represent_image_features(des, codebook)]
    prediction = SIFT_model.predict(feature_vector)
    return prediction[0]

def classify_image_vgg(img):
    img = cv2.resize(img, (224, 224))
    img = img.reshape(-1, 224, 224, 3)
    img = np.array(img)
    preds = vgg_model.predict(img)

    # Convert predictions to one-hot encoding
    one_hot_preds = np.zeros(preds.shape)
    one_hot_preds[np.arange(preds.shape[0]), preds.argmax(axis=1)] = 1
    
    # Get the label from predictions
    label = np.argmax(preds, axis=1)[0]

    return label

def define_extract_model(dims):
    if dims == 4096:
        index = 11
    else:
        index = 13

    output = [ vgg_model.layers[index].output ]
    n_model = Model(inputs=vgg_model.inputs, outputs=output)

    return n_model

def classify_extracted_feature(img,dims):
    model = define_extract_model(dims)

    img = cv2.resize(img, (224, 224))
    img = np.array(img).reshape(-1, 224, 224, 3)
    img = model.predict(img)

    svm_model = joblib.load(f"./VGG8/VGG8_SVM/data/model/{20240520}_SVM_{dims}.joblib")

    predict = svm_model.predict(img)

    return predict[0]


def classify_pca_extract(img, dims, pca_dims):
    pca_name = f"20240520_PCA_" + str(pca_dims) + "_" + str(dims)
    pca_model = joblib.load(f"./VGG8/VGG8_PCA_SVM/data/model/{pca_name}.joblib")

    model_name = f"20240520_SVM_{dims}_{pca_dims}"
    svm_model = joblib.load(f"./VGG8/VGG8_PCA_SVM/data/model/{model_name}.joblib")

    model = define_extract_model(dims)
    img = cv2.resize(img, (224, 224))
    img = np.array(img).reshape(-1, 224, 224, 3)
    img = model.predict(img)

    feature_vector = pca_model.transform(img)

    predict = svm_model.predict(feature_vector)
    
    return predict[0]


