import numpy as np
import pandas as pd

from utils.feature_extractor import FeatureExtractor
from utils.similarity import find_similar_products


EMBEDDINGS_PATH = "embeddings/baseline_embeddings.npy"
METADATA_PATH = "embeddings/metadata.csv"


def main():
    embeddings = np.load(EMBEDDINGS_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    product_ids = metadata["id"].values

    query_index = 0

    query_image = metadata.iloc[
        query_index
    ]["image_path"]

    query_category = metadata.iloc[
        query_index
    ]["articleType"]

    print("\nQuery Image:")
    print(query_image)

    print("Query Category:")
    print(query_category)

    extractor = FeatureExtractor()

    query_embedding = extractor.extract(
        query_image
    )
    results = find_similar_products(
        query_embedding,
        embeddings,
        product_ids,
        top_k=5
    )

    print("\nTop 5 Similar Products:")

    for product_id, score in results:

        product = metadata[
            metadata["id"] == product_id
        ].iloc[0]

        print(
            f"ID: {product_id} | "
            f"Category: {product['articleType']} | "
            f"Similarity: {score:.4f}"
        )


if __name__ == "__main__":
    main()
