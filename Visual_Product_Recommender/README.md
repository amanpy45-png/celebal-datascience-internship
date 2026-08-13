# Visual Product Recommender

An image-based product recommendation system that retrieves visually similar fashion products using deep learning and transfer learning.

The system accepts a product image, extracts deep visual features, and retrieves the most visually similar products from a fashion product dataset. It implements both a ResNet50 baseline and a Siamese Network trained with triplet loss for learning compact image embeddings.

---

## Problem Statement

In e-commerce platforms, users may want to find products that look similar to an item they already have. Traditional keyword-based search cannot always capture visual characteristics such as color, shape, style, and design.

This project addresses this problem by building an image-based recommendation engine that retrieves visually similar fashion products using deep learning embeddings.

---

## Objectives

- Accept a product image as input.
- Extract meaningful visual features using a pretrained CNN.
- Retrieve the Top-K visually similar products.
- Implement a ResNet50-based baseline recommendation system.
- Develop a Siamese Network using triplet loss.
- Compare both approaches using Precision@5 and retrieval time.
- Provide an interactive Streamlit interface for image-based search.

---

## System Architecture

```text
                 Input Product Image
                         |
                         v
                 Image Preprocessing
                         |
                         v
              +-----------------------+
              |   Feature Extraction  |
              +-----------------------+
                   /             \
                  /               \
                 v                 v
        ResNet50 Baseline    Siamese Network
        2048-D Embedding     128-D Embedding
                 |                 |
                 v                 v
              Cosine Similarity / Similarity Search
                         |
                         v
                  Top-K Products
                         |
                         v
                 Streamlit UI
