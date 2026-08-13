import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import image

from utils.feature_extractor import FeatureExtractor


CSV_PATH = "data/subset.csv"
OUTPUT_PATH = "embeddings/baseline_embeddings.npy"
METADATA_PATH = "embeddings/metadata.csv"

BATCH_SIZE = 32


def load_batch(image_paths):
    batch = []

    for path in image_paths:
        try:
            img = image.load_img(
                path,
                target_size=(224, 224)
            )

            img_array = image.img_to_array(img)
            batch.append(img_array)

        except Exception as e:
            print(f"Skipping {path}: {e}")

    return np.array(batch)


def build_embeddings():

    df = pd.read_csv(CSV_PATH)

    extractor = FeatureExtractor()

    model = extractor.model

    all_embeddings = []
    valid_indices = []

    total = len(df)

    print(f"Processing {total} images in batches of {BATCH_SIZE}...\n")

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_df = df.iloc[start:end]

        image_paths = batch_df["image_path"].tolist()

        batch = load_batch(image_paths)

        if len(batch) == 0:
            continue

        from tensorflow.keras.applications.resnet50 import preprocess_input

        batch = preprocess_input(batch)

        embeddings = model.predict(
            batch,
            verbose=0
        )
        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True
        )

        embeddings = embeddings / np.maximum(
            norms,
            1e-10
        )

        all_embeddings.append(embeddings)

        valid_indices.extend(
            batch_df.index[:len(batch)].tolist()
        )

        processed = min(
            end,
            total
        )

        print(
            f"Processed {processed}/{total}"
        )

    embeddings = np.vstack(
        all_embeddings
    )

    metadata = df.iloc[
        valid_indices
    ].reset_index(drop=True)

    os.makedirs(
        "embeddings",
        exist_ok=True
    )

    np.save(
        OUTPUT_PATH,
        embeddings
    )

    metadata.to_csv(
        METADATA_PATH,
        index=False
    )


    print("Embedding generation complete!")

    print(
        "Embedding shape:",
        embeddings.shape
    )
    print(
        "Saved:",
        OUTPUT_PATH
    )
    print(
        "Saved:",
        METADATA_PATH
    )


if __name__ == "__main__":
    build_embeddings()