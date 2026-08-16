import os
import random
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam

from utils.siamese import SiameseNetwork, build_triplet_model, triplet_loss


CSV_PATH = "data/subset.csv"
MODEL_PATH = "models/siamese_embedding.keras"

IMG_SIZE = (224, 224)
TRIPLETS = 1000
BATCH_SIZE = 16
EPOCHS = 2

def load_image(path):
    img = load_img(
        path,
        target_size=IMG_SIZE
    )
    img = img_to_array(img)
    return img

def create_triplets(df, num_triplets):
    triplets = []
    categories = df["articleType"].unique()
    grouped = {
        category: group
        for category, group in df.groupby("articleType")
    }

    for _ in range(num_triplets):
        category = random.choice(categories)
        category_df = grouped[category]


        if len(category_df) < 2:
            continue

        anchor, positive = category_df.sample(
            2,
            replace=False
        )["image_path"].tolist()


        negative_category = random.choice(
            [
                c for c in categories
                if c != category
            ]
        )

        negative = grouped[
            negative_category
        ].sample(
            1
        )["image_path"].iloc[0]

        triplets.append(
            (
                anchor,
                positive,
                negative
            )
        )

    return triplets


def main():

    os.makedirs(
        "models",
        exist_ok=True
    )

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        f"Dataset: {len(df)} images"
    )

    print(
        "Creating triplets..."
    )

    triplets = create_triplets(
        df,
        TRIPLETS
    )

    print(
        f"Triplets created: {len(triplets)}"
    )

    siamese = SiameseNetwork(
        embedding_dim=128
    )

    embedding_model = siamese.get_model()
    model = build_triplet_model(
        embedding_model
    )

    model.compile(
        optimizer=Adam(
            learning_rate=1e-4
        ),
        loss=triplet_loss(
            margin=0.2
        )
    )

    anchors = []
    positives = []
    negatives = []

    print(
        "Loading triplet images..."
    )

    for i, (anchor, positive, negative) in enumerate(triplets):

        try:

            anchors.append(
                load_image(anchor)
            )

            positives.append(
                load_image(positive)
            )

            negatives.append(
                load_image(negative)
            )

        except Exception as e:

            print(
                f"Skipping triplet {i}: {e}"
            )

    anchors = np.array(anchors)
    positives = np.array(positives)
    negatives = np.array(negatives)


    y = np.zeros(
        len(anchors)
    )

    print(
        f"Training on {len(anchors)} triplets..."
    )

    model.fit(
        [
            anchors,
            positives,
            negatives
        ],
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        shuffle=True
    )

    embedding_model.save(
        MODEL_PATH
    )

    print(
        "\n=============================="
    )

    print(
        "Siamese training complete!"
    )

    print(
        f"Saved model: {MODEL_PATH}"
    )

    print(
        "Embedding dimension: 128"
    )


if __name__ == "__main__":
    main()
