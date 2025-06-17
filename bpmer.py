from pathlib import Path
from typing import Tuple, Dict

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras



def load_bmp(path: str | Path) -> np.ndarray:
   
    img = Image.open(path).convert("L")
    return np.asarray(img, np.float32)[..., None]



class PneumoniaScanner:
    CLASS_NAMES = ["Normal", "Bacterial Pneumonia", "Viral Pneumonia"]

    _CUSTOM_OBJECTS = {
        "preprocess_input": tf.keras.applications.resnet.preprocess_input
    }

    def __init__(self, model_path: str | Path):
        self.model = keras.models.load_model(model_path,
                                             custom_objects=self._CUSTOM_OBJECTS)  

    def predict(self, bmp_path):
        img = load_bmp(bmp_path)                 
        img = tf.image.resize(img, (224, 224))
        img = tf.image.grayscale_to_rgb(img)     
        img = tf.keras.applications.resnet.preprocess_input(img)
        probs = self.model(tf.expand_dims(img, 0), training=False)[0].numpy()
        return int(np.argmax(probs)), probs

    def visualize_prediction(self, bmp_path: str | Path, save_path: str | Path):
        img = load_bmp(bmp_path).squeeze()
        plt.imsave(save_path, img, cmap="gray")
