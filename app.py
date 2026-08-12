"""
Fashion Recommendation System
=============================
Uses a pre-trained ResNet18 (Torchvision) to extract image embeddings and
Spotify's Annoy library for fast approximate nearest-neighbor search.

Dataset: Fashion-MNIST (10 clothing categories, ~60k training images).
Run:     python app.py
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from annoy import AnnoyIndex
except ImportError as exc:
    raise ImportError(
        "The 'annoy' package is required. Install it with:\n"
        "  pip install annoy\n\n"
        "On Windows with Python 3.13, you may need Microsoft C++ Build Tools:\n"
        "  https://visualstudio.microsoft.com/visual-cpp-build-tools/\n"
        "Alternatively, use Python 3.11/3.12 or: conda install -c conda-forge python-annoy"
    ) from exc
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
SAMPLE_SIZE = 2_000          # Use a subset for a quick demo (set None for full set)
EMBEDDING_DIM = 512          # ResNet18 penultimate layer output size
ANNOY_NUM_TREES = 10         # More trees = better recall, slower build
TOP_K = 5                    # Number of similar items to return
QUERY_INDEX = 42             # Index of the image to use as the query
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fashion-MNIST class labels (useful when printing results)
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


# ---------------------------------------------------------------------------
# 1. Download & prepare the fashion image dataset
# ---------------------------------------------------------------------------
def download_fashion_dataset(data_dir: Path) -> datasets.FashionMNIST:
    """
    Download Fashion-MNIST via Torchvision and return the training split.
    Images are 28×28 grayscale; we convert them to RGB later for ResNet18.
    """
    data_dir.mkdir(parents=True, exist_ok=True)

    # Only normalization here — resizing/conversion happens in the embedding step
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = datasets.FashionMNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )
    print(f"Downloaded Fashion-MNIST: {len(dataset):,} training images.")
    return dataset


def subset_dataset(dataset, sample_size: int | None):
    """Optionally limit the dataset size so the demo runs quickly."""
    if sample_size is None or sample_size >= len(dataset):
        return dataset
    indices = list(range(sample_size))
    print(f"Using a subset of {sample_size:,} images for faster processing.")
    return Subset(dataset, indices)


# ---------------------------------------------------------------------------
# 2. Load pre-trained ResNet18 and strip the classification head
# ---------------------------------------------------------------------------
def load_feature_extractor() -> nn.Module:
    """
    Load ResNet18 pre-trained on ImageNet and replace the final FC layer
    with an identity so the model outputs a 512-dimensional feature vector.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    backbone = models.resnet18(weights=weights)

    # Remove the final classification layer → output shape: (batch, 512)
    backbone.fc = nn.Identity()
    backbone.eval()
    backbone.to(DEVICE)

    print(f"Loaded ResNet18 feature extractor on {DEVICE}.")
    return backbone


def build_resnet_transform():
    """
    Preprocessing pipeline expected by ImageNet-trained ResNet18:
    resize to 224×224, convert grayscale→RGB, normalize with ImageNet stats.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


# ---------------------------------------------------------------------------
# 3. Extract feature embeddings for every image in the dataset
# ---------------------------------------------------------------------------
@torch.no_grad()
def extract_embeddings(
    model: nn.Module,
    dataset,
    resnet_transform,
) -> tuple[np.ndarray, list[int]]:
    """
    Pass each image through ResNet18 and collect L2-normalized embeddings.
    Normalization makes cosine similarity equivalent to Euclidean distance,
    which works well with Annoy's angular metric.
    """
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    all_embeddings: list[np.ndarray] = []
    all_labels: list[int] = []

    for batch_images, batch_labels in loader:
        # Fashion-MNIST tensors are (B, 1, 28, 28) — convert for ResNet input
        batch_tensors = []
        for img_tensor in batch_images:
            # Denormalize from [-1, 1] back to [0, 1] for PIL conversion
            img_np = img_tensor.squeeze().numpy()
            img_np = (img_np * 0.5 + 0.5).clip(0, 1)
            pil_img = Image.fromarray((img_np * 255).astype(np.uint8), mode="L")
            batch_tensors.append(resnet_transform(pil_img))

        batch = torch.stack(batch_tensors).to(DEVICE)
        features = model(batch)                          # (B, 512)
        features = features.cpu().numpy()

        # L2-normalize each embedding vector
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        features = features / (norms + 1e-8)

        all_embeddings.append(features)
        all_labels.extend(batch_labels.tolist())

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    print(f"Extracted {embeddings.shape[0]:,} embeddings of dim {embeddings.shape[1]}.")
    return embeddings, all_labels


# ---------------------------------------------------------------------------
# 4. Build an Annoy index for fast similarity search
# ---------------------------------------------------------------------------
def build_annoy_index(embeddings: np.ndarray) -> AnnoyIndex:
    """
    Create an AnnoyIndex using angular distance (ideal for normalized vectors).
    Each item i in the index corresponds to image i in the dataset.
    """
    index = AnnoyIndex(EMBEDDING_DIM, metric="angular")

    for i, vector in enumerate(embeddings):
        index.add_item(i, vector.tolist())

    index.build(ANNOY_NUM_TREES)
    print(f"Built Annoy index with {len(embeddings):,} items and {ANNOY_NUM_TREES} trees.")
    return index


# ---------------------------------------------------------------------------
# 5. Query: find the top-K most similar images to a given input
# ---------------------------------------------------------------------------
def find_similar(
    index: AnnoyIndex,
    query_vector: np.ndarray,
    labels: list[int],
    query_idx: int,
    top_k: int = TOP_K,
) -> list[tuple[int, float, str]]:
    """
    Return the top-K nearest neighbors for query_vector.
    Skips the query image itself if it appears in the results.
    """
    # Request one extra neighbor in case the query matches itself
    neighbor_indices, distances = index.get_nns_by_vector(
        query_vector.tolist(),
        n=top_k + 1,
        include_distances=True,
    )

    results: list[tuple[int, float, str]] = []
    for idx, dist in zip(neighbor_indices, distances):
        if idx == query_idx:
            continue
        # Convert angular distance to cosine similarity for readability
        cosine_sim = 1.0 - (dist ** 2) / 2.0
        results.append((idx, cosine_sim, CLASS_NAMES[labels[idx]]))
        if len(results) >= top_k:
            break

    return results


def print_recommendations(
    query_idx: int,
    query_label: str,
    recommendations: list[tuple[int, float, str]],
) -> None:
    """Pretty-print the query info and its top similar items."""
    print("\n" + "=" * 60)
    print("FASHION RECOMMENDATION RESULTS")
    print("=" * 60)
    print(f"Query image index : {query_idx}")
    print(f"Query category    : {query_label}")
    print("-" * 60)
    print(f"{'Rank':<6} {'Index':<8} {'Similarity':<14} {'Category'}")
    print("-" * 60)

    for rank, (idx, similarity, category) in enumerate(recommendations, start=1):
        print(f"{rank:<6} {idx:<8} {similarity:.4f}{'':<6} {category}")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    # Step 1 — Download dataset
    dataset = download_fashion_dataset(DATA_DIR)
    dataset = subset_dataset(dataset, SAMPLE_SIZE)

    # Step 2 — Load feature extractor
    model = load_feature_extractor()
    resnet_transform = build_resnet_transform()

    # Step 3 — Extract embeddings
    embeddings, labels = extract_embeddings(model, dataset, resnet_transform)

    # Step 4 — Build Annoy index
    index = build_annoy_index(embeddings)

    # Step 5 — Query with a sample image and print top-5 recommendations
    query_idx = min(QUERY_INDEX, len(labels) - 1)
    query_vector = embeddings[query_idx]
    query_label = CLASS_NAMES[labels[query_idx]]

    recommendations = find_similar(index, query_vector, labels, query_idx)
    print_recommendations(query_idx, query_label, recommendations)


if __name__ == "__main__":
    main()
