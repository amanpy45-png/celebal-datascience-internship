import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity


def precision_at_k(
    retrieved_categories,
    query_category,
    k
):
    """
    Precision@K:
    Fraction of retrieved products in top-K
    that belong to the same category as the query.
    """

    retrieved_categories = retrieved_categories[:k]

    if not retrieved_categories:
        return 0.0

    relevant = sum(
        category == query_category
        for category in retrieved_categories
    )

    return relevant / len(retrieved_categories)


def recall_at_k(
    retrieved_categories,
    query_category,
    total_relevant,
    k
):
    """
    Recall@K:
    Fraction of all relevant products that
    were retrieved in the top-K results.
    """

    retrieved_categories = retrieved_categories[:k]

    if total_relevant == 0:
        return 0.0

    relevant_retrieved = sum(
        category == query_category
        for category in retrieved_categories
    )

    return relevant_retrieved / total_relevant


def measure_inference_time(
    similarity_function,
    query_embedding,
    product_embeddings,
    product_ids,
    top_k=5
):
    """
    Measure retrieval/inference time.
    """

    start = time.perf_counter()

    results = similarity_function(
        query_embedding,
        product_embeddings,
        product_ids,
        top_k
    )

    end = time.perf_counter()

    inference_time = end - start

    return results, inference_time