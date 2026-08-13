import os
import pandas as pd


CSV_PATH = "data/fashion-dataset/styles.csv"
IMAGE_DIR = "data/fashion-dataset/images"
OUTPUT_CSV = "data/subset.csv"

CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Kurtas",
    "Tops",
    "Handbags",
    "Heels"
]

SAMPLES_PER_CATEGORY = 250


def create_subset():
    df = pd.read_csv(
        CSV_PATH,
        on_bad_lines="skip"
    )

    subset_parts = []

    print("Creating dataset subset...\n")

    for category in CATEGORIES:

        category_df = df[
            df["articleType"] == category
        ].copy()

        category_df["image_path"] = category_df["id"].apply(
            lambda x: os.path.join(
                IMAGE_DIR,
                f"{int(x)}.jpg"
            )
        )

        category_df = category_df[
            category_df["image_path"].apply(
                os.path.exists
            )
        ]

        sample_size = min(
            SAMPLES_PER_CATEGORY,
            len(category_df)
        )

        category_df = category_df.sample(
            n=sample_size,
            random_state=42
        )

        subset_parts.append(category_df)

        print(
            f"{category}: {sample_size} images"
        )


    subset = pd.concat(
        subset_parts,
        ignore_index=True
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    subset.to_csv(
        OUTPUT_CSV,
        index=False
    )

    print("\n-----------------------------")
    print("SUBSET CREATED")
    print("-----------------------------")
    print(f"Total images: {len(subset)}")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nCategory distribution:")
    print(
        subset["articleType"].value_counts()
    )

    return subset


if __name__ == "__main__":
    create_subset()