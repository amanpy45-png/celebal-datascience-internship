import time
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from utils.feature_extractor import FeatureExtractor
from utils.similarity import find_similar_products


EMBEDDINGS_PATH = "embeddings/baseline_embeddings.npy"
METADATA_PATH = "embeddings/metadata.csv"
SIAMESE_MODEL_PATH = "models/siamese_embedding.keras"

K = 5
NUM_QUERIES = 50

def precision_at_k(retrieved_categories, query_category, k):
    retrieved = retrieved_categories[:k]
    if not retrieved:
        return 0.0

    relevant = sum(
        category == query_category
        for category in retrieved
    )
    return relevant / k


def evaluate_baseline(df, embeddings):

    print("\nEvaluating Baseline...")

    scores = []
    times = []

    product_ids = df["id"].values

    for i in range(min(NUM_QUERIES, len(df))):

        query_embedding = embeddings[i]

        start = time.perf_counter()

        results = find_similar_products(
            query_embedding,
            embeddings,
            product_ids,
            top_k=K + 1
        )

        elapsed = time.perf_counter() - start

        results = [
            r for r in results
            if r[0] != product_ids[i]
        ][:K]

        categories = []

        for product_id, _ in results:

            row = df[
                df["id"] == product_id
            ].iloc[0]

            categories.append(
                row["articleType"]
            )

        score = precision_at_k(
            categories,
            df.iloc[i]["articleType"],
            K
        )

        scores.append(score)
        times.append(elapsed)

    return np.mean(scores), np.mean(times)


def evaluate_siamese(df):

    print("\nEvaluating Siamese model...")

    model = load_model(
    SIAMESE_MODEL_PATH,
    compile=False,
    safe_mode=False
    )

    product_ids = df["id"].values

    all_embeddings = []

    start_embedding = time.perf_counter()

    for start in range(
        0,
        len(df),
        32
    ):

        batch_df = df.iloc[
            start:start + 32
        ]

        images = []

        for path in batch_df["image_path"]:

            from tensorflow.keras.preprocessing.image import load_img
            from tensorflow.keras.preprocessing.image import img_to_array
            from tensorflow.keras.applications.resnet50 import preprocess_input

            img = load_img(
                path,
                target_size=(224, 224)
            )

            img = img_to_array(img)

            images.append(img)

        images = preprocess_input(
            np.array(images)
        )

        batch_embeddings = model.predict(
            images,
            verbose=0
        )

        all_embeddings.append(
            batch_embeddings
        )

    siamese_embeddings = np.vstack(
        all_embeddings
    )

    embedding_time = (
        time.perf_counter() -
        start_embedding
    )

    scores = []
    times = []

    for i in range(
        min(NUM_QUERIES, len(df))
    ):

        query_embedding = siamese_embeddings[i]

        start = time.perf_counter()

        results = find_similar_products(
            query_embedding,
            siamese_embeddings,
            product_ids,
            top_k=K + 1
        )

        elapsed = time.perf_counter() - start

        results = [
            r for r in results
            if r[0] != product_ids[i]
        ][:K]

        categories = []

        for product_id, _ in results:

            row = df[
                df["id"] == product_id
            ].iloc[0]

            categories.append(
                row["articleType"]
            )

        score = precision_at_k(
            categories,
            df.iloc[i]["articleType"],
            K
        )

        scores.append(score)
        times.append(elapsed)

    return (
        np.mean(scores),
        np.mean(times),
        embedding_time
    )


def main():

    df = pd.read_csv(
        METADATA_PATH
    )

    baseline_embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Evaluating on {len(df)} products"
    )

    baseline_precision, baseline_time = (
        evaluate_baseline(
            df,
            baseline_embeddings
        )
    )

    siamese_precision, siamese_time, embedding_time = (
        evaluate_siamese(df)
    )


    print("MODEL COMPARISON")


    print(
        f"\nBaseline ResNet50"
    )

    print(
        f"Precision@{K}: "
        f"{baseline_precision:.4f}"
    )

    print(
        f"Retrieval time: "
        f"{baseline_time:.6f} sec"
    )

    print(
        f"\nSiamese Network"
    )

    print(
        f"Precision@{K}: "
        f"{siamese_precision:.4f}"
    )

    print(
        f"Retrieval time: "
        f"{siamese_time:.6f} sec"
    )

    print(
        f"Embedding generation time: "
        f"{embedding_time:.2f} sec"
    )

    print("\n===================================")

    improvement = (
        (
            siamese_precision -
            baseline_precision
        )
        /
        max(baseline_precision, 1e-10)
    ) * 100

    print(
        f"Precision improvement: "
        f"{improvement:.2f}%"
    )


if __name__ == "__main__":
    main()
