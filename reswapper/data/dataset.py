import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class VGGFace2Dataset(Dataset):
    """Dataset for identity-grouped face images (VGGFace2-HQ).

    Loads pre-processed face data: aligned images, identity embeddings,
    face parsing maps, and optionally pre-encoded VAE latents.

    Supports two sampling modes per item:
    - Same-identity pair (~50%): two images of the same person for reconstruction
    - Cross-identity pair (~50%): two different people for swapping

    Expected directory structure (from preprocessing):
        dataset_dir/
            n000001/
                00001.jpg          # aligned face 512x512
                00001_emb.npy      # 512-D identity embedding
                00001_parse.npy    # 19-class parsing map
                00001_latent.npy   # (optional) VAE-encoded latent
                00002.jpg
                00002_emb.npy
                ...
            n000002/
                ...
    """

    def __init__(
        self,
        dataset_dir: str,
        same_identity_ratio: float = 0.5,
        resolution: int = 512,
        transform=None,
    ):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.same_identity_ratio = same_identity_ratio
        self.resolution = resolution
        self.transform = transform

        # Build index: identity_id -> list of image stems
        self.identities = {}
        self.all_samples = []

        if self.dataset_dir.exists():
            for identity_dir in sorted(self.dataset_dir.iterdir()):
                if not identity_dir.is_dir():
                    continue
                identity_id = identity_dir.name
                stems = []
                for img_path in sorted(identity_dir.glob("*.jpg")) + sorted(identity_dir.glob("*.png")):
                    stem = img_path.stem
                    if stem.endswith(("_emb", "_parse", "_latent")):
                        continue
                    stems.append(stem)
                if stems:
                    self.identities[identity_id] = stems
                    for stem in stems:
                        self.all_samples.append((identity_id, stem))

        self.identity_list = list(self.identities.keys())

    def __len__(self) -> int:
        return len(self.all_samples)

    def _load_sample(self, identity_id: str, stem: str) -> dict:
        """Load a single preprocessed sample."""
        base_dir = self.dataset_dir / identity_id

        # Load image
        img_path = base_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = base_dir / f"{stem}.png"

        import cv2
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.resolution:
            img = cv2.resize(img, (self.resolution, self.resolution))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # [3, H, W] in [0, 1]

        # Load embedding
        emb_path = base_dir / f"{stem}_emb.npy"
        if emb_path.exists():
            embedding = torch.from_numpy(np.load(str(emb_path))).float().squeeze()
        else:
            embedding = torch.zeros(512)

        # Load parsing map
        parse_path = base_dir / f"{stem}_parse.npy"
        if parse_path.exists():
            parsing = torch.from_numpy(np.load(str(parse_path))).float()
        else:
            # Default: all skin
            parsing = torch.zeros(19, self.resolution, self.resolution)
            parsing[1] = 1.0

        # Load VAE latent if available
        latent_path = base_dir / f"{stem}_latent.npy"
        if latent_path.exists():
            latent = torch.from_numpy(np.load(str(latent_path))).float()
        else:
            latent = None

        return {
            "image": img,
            "embedding": embedding,
            "parsing": parsing,
            "latent": latent,
        }

    def __getitem__(self, idx: int) -> dict:
        target_identity, target_stem = self.all_samples[idx]

        # Decide same-identity or cross-identity
        is_same_identity = random.random() < self.same_identity_ratio

        if is_same_identity and len(self.identities[target_identity]) > 1:
            # Pick a different image of the same person as source
            source_stem = random.choice(
                [s for s in self.identities[target_identity] if s != target_stem]
            )
            source_identity = target_identity
        else:
            # Pick a random person as source
            source_identity = random.choice(self.identity_list)
            source_stem = random.choice(self.identities[source_identity])
            is_same_identity = source_identity == target_identity

        target_data = self._load_sample(target_identity, target_stem)
        source_data = self._load_sample(source_identity, source_stem)

        # Apply augmentation to target image
        if self.transform is not None:
            target_data["image"] = self.transform(target_data["image"])

        result = {
            "target_image": target_data["image"],
            "target_parsing": target_data["parsing"],
            "source_embedding": source_data["embedding"],
            "is_same_identity": torch.tensor(is_same_identity, dtype=torch.bool),
        }

        # Include latents if available
        if target_data["latent"] is not None:
            result["target_latent"] = target_data["latent"]

        return result


class FFHQDataset(Dataset):
    """Simple FFHQ dataset for cross-identity swapping only (no identity labels).

    Expected structure:
        dataset_dir/
            00000.jpg
            00000_emb.npy
            00000_parse.npy
            00000_latent.npy  (optional)
            ...
    """

    def __init__(self, dataset_dir: str, resolution: int = 512, transform=None):
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.resolution = resolution
        self.transform = transform

        self.stems = []
        if self.dataset_dir.exists():
            for img_path in sorted(self.dataset_dir.glob("*.jpg")) + sorted(self.dataset_dir.glob("*.png")):
                stem = img_path.stem
                if stem.endswith(("_emb", "_parse", "_latent")):
                    continue
                self.stems.append(stem)

    def __len__(self) -> int:
        return len(self.stems)

    def _load_sample(self, stem: str) -> dict:
        import cv2
        img_path = self.dataset_dir / f"{stem}.jpg"
        if not img_path.exists():
            img_path = self.dataset_dir / f"{stem}.png"

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[0] != self.resolution:
            img = cv2.resize(img, (self.resolution, self.resolution))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        emb_path = self.dataset_dir / f"{stem}_emb.npy"
        embedding = torch.from_numpy(np.load(str(emb_path))).float().squeeze() if emb_path.exists() else torch.zeros(512)

        parse_path = self.dataset_dir / f"{stem}_parse.npy"
        if parse_path.exists():
            parsing = torch.from_numpy(np.load(str(parse_path))).float()
        else:
            parsing = torch.zeros(19, self.resolution, self.resolution)
            parsing[1] = 1.0

        latent_path = self.dataset_dir / f"{stem}_latent.npy"
        latent = torch.from_numpy(np.load(str(latent_path))).float() if latent_path.exists() else None

        return {"image": img, "embedding": embedding, "parsing": parsing, "latent": latent}

    def __getitem__(self, idx: int) -> dict:
        target_stem = self.stems[idx]
        source_idx = random.randint(0, len(self.stems) - 1)
        source_stem = self.stems[source_idx]

        target_data = self._load_sample(target_stem)
        source_data = self._load_sample(source_stem)

        if self.transform:
            target_data["image"] = self.transform(target_data["image"])

        result = {
            "target_image": target_data["image"],
            "target_parsing": target_data["parsing"],
            "source_embedding": source_data["embedding"],
            "is_same_identity": torch.tensor(idx == source_idx, dtype=torch.bool),
        }
        if target_data["latent"] is not None:
            result["target_latent"] = target_data["latent"]
        return result
