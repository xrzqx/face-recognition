import tensorflow as tf
import numpy as np
import cv2

import os
import zipfile
import gdown

url="https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip"
if os.path.isfile("VGGFace2_DeepFace_weights_val-0.9034.h5") != True:
    print("VGGFace2_DeepFace_weights_val-0.9034.h5 will be downloaded...")
    output = "VGGFace2_DeepFace_weights_val-0.9034.h5.zip"
    gdown.download(url, output, quiet=False)
    # unzip VGGFace2_DeepFace_weights_val-0.9034.h5.zip
    with zipfile.ZipFile(output, "r") as zip_ref:
        zip_ref.extractall(".")

model = tf.keras.Sequential()
model.add(tf.keras.layers.Convolution2D(32, (11, 11), activation='relu', name='C1', input_shape=(152, 152, 3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same', name='M2'))
model.add(tf.keras.layers.Convolution2D(16, (9, 9), activation='relu', name='C3'))
model.add(tf.keras.layers.LocallyConnected2D(16, (9, 9), activation='relu', name='L4'))
model.add(tf.keras.layers.LocallyConnected2D(16, (7, 7), strides=2, activation='relu', name='L5') )
model.add(tf.keras.layers.LocallyConnected2D(16, (5, 5), activation='relu', name='L6'))
model.add(tf.keras.layers.Flatten(name='F0'))
model.add(tf.keras.layers.Dense(4096, activation='relu', name='F7'))
model.add(tf.keras.layers.Dropout(rate=0.5, name='D0'))
model.add(tf.keras.layers.Dense(8631, activation='softmax', name='F8'))

model.load_weights("VGGFace2_DeepFace_weights_val-0.9034.h5")

deepface_model = tf.keras.Model(inputs=model.layers[0].input, outputs=model.layers[-3].output)

def l2_normalize(x):
    return x / np.sqrt(np.sum(np.multiply(x, x)))
 
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance

def preprocess_image(image_opencv):

    img = cv2.resize(image_opencv, (152, 152))
    img = tf.keras.preprocessing.image.array_to_img(img)
    # add channels first
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.imagenet_utils.preprocess_input(img)
    return img


def get_embedding(image):
    img1_embedding = deepface_model.predict(preprocess_image(image))[0,:]
    return img1_embedding

def face_verify(img1_embedding,img2_embedding):
    euclidean_l2_distance = findEuclideanDistance(l2_normalize(img1_embedding)
    , l2_normalize(img2_embedding))

    if euclidean_l2_distance <= 0.55:
        print("verified... they are same person")
        verified = True
        return verified
    else:
        print("unverified! they are not same person!")
        verified = False
        return verified