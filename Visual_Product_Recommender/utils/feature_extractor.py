import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image


class FeatureExtractor:
    def __init__(self):
        self.model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

    def extract(self, image_path):
        img = image.load_img(
            image_path,
            target_size=(224, 224)
        )

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        img_array = preprocess_input(img_array)

        embedding = self.model.predict(
            img_array,
            verbose=0
        )[0]

        embedding = embedding / np.linalg.norm(embedding)

        return embedding