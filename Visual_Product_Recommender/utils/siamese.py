import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input


class SiameseNetwork:

    def __init__(self, embedding_dim=128):

        base_model = ResNet50(
            weights="imagenet",
            include_top=False,
            pooling="avg"
        )

        for layer in base_model.layers:
            layer.trainable = False

        for layer in base_model.layers[-20:]:
            layer.trainable = True

        inputs = layers.Input(
            shape=(224, 224, 3)
        )

        x = preprocess_input(inputs)

        x = base_model(x)

        x = layers.Dense(
            512,
            activation="relu"
        )(x)

        x = layers.Dropout(0.3)(x)

        embeddings = layers.Dense(
            embedding_dim
        )(x)

        embeddings = layers.UnitNormalization(
            axis=1
            )(embeddings)

        self.embedding_model = Model(
            inputs,
            embeddings,
            name="siamese_embedding_model"
        )

    def get_model(self):
        return self.embedding_model


def triplet_loss(margin=0.2):

    def loss(y_true, y_pred):

        anchor = y_pred[:, 0:128]
        positive = y_pred[:, 128:256]
        negative = y_pred[:, 256:384]

        positive_distance = tf.reduce_sum(
            tf.square(anchor - positive),
            axis=1
        )

        negative_distance = tf.reduce_sum(
            tf.square(anchor - negative),
            axis=1
        )

        loss_value = tf.maximum(
            positive_distance -
            negative_distance +
            margin,
            0.0
        )

        return tf.reduce_mean(loss_value)

    return loss


def build_triplet_model(embedding_model):

    anchor_input = layers.Input(
        shape=(224, 224, 3),
        name="anchor"
    )

    positive_input = layers.Input(
        shape=(224, 224, 3),
        name="positive"
    )

    negative_input = layers.Input(
        shape=(224, 224, 3),
        name="negative"
    )

    anchor_embedding = embedding_model(
        anchor_input
    )

    positive_embedding = embedding_model(
        positive_input
    )

    negative_embedding = embedding_model(
        negative_input
    )

    output = layers.Concatenate(
        axis=1
    )([
        anchor_embedding,
        positive_embedding,
        negative_embedding
    ])

    model = Model(
        inputs=[
            anchor_input,
            positive_input,
            negative_input
        ],
        outputs=output,
        name="triplet_siamese_network"
    )

    return model