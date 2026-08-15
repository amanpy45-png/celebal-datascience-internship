import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tempfile

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input

from utils.feature_extractor import FeatureExtractor
from utils.similarity import find_similar_products


BASELINE_EMBEDDINGS = "embeddings/baseline_embeddings.npy"
METADATA_PATH = "embeddings/metadata.csv"
SIAMESE_MODEL = "models/siamese_embedding.keras"

IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="Visual Product Recommender",
    layout="wide"
)
st.title("Visual Product Recommendation System")

st.write(
    "Upload a product image and retrieve visually similar "
    "products using deep learning."
)

@st.cache_data
def load_baseline_data():

    embeddings = np.load(
        BASELINE_EMBEDDINGS
    )

    metadata = pd.read_csv(
        METADATA_PATH
    )
    return embeddings, metadata


@st.cache_resource
def load_baseline_model():

    return FeatureExtractor()

@st.cache_resource
def load_siamese_model():

    return load_model(
        SIAMESE_MODEL,
        compile=False,
        safe_mode=False
    )
st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Select Model",
    [
        "ResNet50 Baseline",
        "Siamese Network"
    ],
    index=0
)
top_k = st.sidebar.slider(
    "Number of Recommendations",
    3,
    10,
    5
)

uploaded_file = st.file_uploader(
    "Upload Product Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        st.subheader("Query Image")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader(
            f"Model: {model_choice}"
        )

        if st.button(
            "Find Similar Products",
            use_container_width=True
        ):

            with st.spinner(
                "Finding similar products..."
            ):

                embeddings, metadata = (
                    load_baseline_data()
                )

                if model_choice == "ResNet50 Baseline":

                    model = load_baseline_model()

                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".jpg"
                    ) as temp:

                        image.save(
                            temp.name
                        )

                        temp_path = temp.name

                    try:

                        start = time.perf_counter()

                        query_embedding = (
                            model.extract(
                                temp_path
                            )
                        )

                        results = (
                            find_similar_products(
                                query_embedding,
                                embeddings,
                                metadata["id"].values,
                                top_k=top_k + 1
                            )
                        )

                        elapsed = (
                            time.perf_counter()
                            - start
                        )

                    finally:

                        os.remove(
                            temp_path
                        )

                else:

                    model = load_siamese_model()

                    img = image.resize(
                        IMAGE_SIZE
                    )

                    img_array = img_to_array(
                        img
                    )

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    img_array = preprocess_input(
                        img_array
                    )

                    start = time.perf_counter()

                    query_embedding = (
                        model.predict(
                            img_array,
                            verbose=0
                        )[0]
                    )
                    siamese_embeddings = []

                    for start_idx in range(
                        0,
                        len(metadata),
                        32
                    ):

                        batch_paths = metadata[
                            "image_path"
                        ].iloc[
                            start_idx:start_idx + 32
                        ]

                        batch_images = []

                        for path in batch_paths:

                            img = load_img(
                                path,
                                target_size=IMAGE_SIZE
                            )

                            img = img_to_array(
                                img
                            )

                            batch_images.append(
                                img
                            )

                        batch_images = preprocess_input(
                            np.array(
                                batch_images
                            )
                        )

                        batch_embeddings = (
                            model.predict(
                                batch_images,
                                verbose=0
                            )
                        )

                        siamese_embeddings.append(
                            batch_embeddings
                        )

                    siamese_embeddings = np.vstack(
                        siamese_embeddings
                    )

                    results = (
                        find_similar_products(
                            query_embedding,
                            siamese_embeddings,
                            metadata["id"].values,
                            top_k=top_k + 1
                        )
                    )

                    elapsed = (
                        time.perf_counter()
                        - start
                    )
            results = [
                result
                for result in results
                if result[0]
                != metadata.iloc[0]["id"]
            ][:top_k]

            st.success(
                f"Found {len(results)} similar products "
                f"in {elapsed:.3f} seconds"
            )

            st.subheader(
                f"Top {len(results)} Recommendations"
            )

            cols = st.columns(
                len(results)
            )

            for col, (product_id, score) in zip(
                cols,
                results
            ):

                row = metadata[
                    metadata["id"] == product_id
                ].iloc[0]

                with col:

                    st.image(
                        row["image_path"],
                        use_container_width=True
                    )

                    st.write(
                        f"**{row['articleType']}**"
                    )

                    st.caption(
                        f"Similarity: {score:.3f}"
                    )

                    st.caption(
                        f"Product ID: {product_id}"
                    )

st.divider()

st.subheader("Model Evaluation")

st.markdown(
    """
| Model | Precision@5 | Retrieval Time |
|---|---:|---:|
| ResNet50 Baseline | **86.8%** | 0.0242 sec |
| Siamese Network | **84.8%** | **0.00116 sec** |

**Baseline:** 2048-dimensional ResNet50 embeddings.

**Siamese:** 128-dimensional embeddings learned using triplet loss.

The Siamese model provides a significantly smaller embedding
representation and faster similarity retrieval, while the
baseline achieved slightly higher Precision@5 in this experiment.
"""
)
