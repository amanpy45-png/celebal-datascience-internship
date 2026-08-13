import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def find_similar_products(
    query_embedding,
    product_embeddings,
    product_ids,
    top_k=5
):
    """
    Find the top-K visually similar products.

    Parameters:
        query_embedding: Embedding of the uploaded image.
        product_embeddings: Embeddings of all products.
        product_ids: IDs corresponding to product embeddings.
        top_k: Number of recommendations.

    Returns:
        List of (product_id, similarity_score).
    """

    query_embedding = np.array(query_embedding).reshape(1, -1)

    similarities = cosine_similarity(
        query_embedding,
        product_embeddings
    )[0]

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append(
            (
                product_ids[idx],
                float(similarities[idx])
            )
        )

    return results