# Histopathology Dataset Storage & Pipeline Directory Structure

This directory houses histopathology image datasets, processed artifacts, and metadata manifests used across the machine learning lifecycle.

## Directory Structure

```
data/
├── raw/                  # (Git-ignored) Original, unmodified datasets from sources
│   ├── pcam/             # PatchCamelyon (PCam) raw patches / archives
│   ├── breakhis/         # BreaKHis breast cancer histopathology images
│   └── external/         # Additional external histopathology sources
│
├── interim/              # (Git-ignored) Validated, sanitized intermediate datasets
│   └── validated/        # Files that passed image integrity and deduplication checks
│
├── processed/            # (Git-ignored) Final split datasets ready for Phase 3 training
│   ├── train/            # Training split
│   ├── val/              # Validation split
│   └── test/             # Held-out test split
│
└── metadata/             # (Git-tracked) Schemas, registries, mappings, and manifests
    ├── dataset_registry.json   # Registry of supported dataset sources and parameters
    ├── class_mapping.json      # Source-to-canonical label definitions
    └── manifests/              # Generated JSONL and CSV dataset manifests
```

## Storage Policy & Git Tracking Rules

- **Raw / Interim / Processed Data**: Large image files (`.png`, `.jpg`, `.tif`, `.h5`, etc.) must **never** be committed to Git. These directories are ignored in `.gitignore`.
- **Metadata**: Registry files (`dataset_registry.json`), class mappings (`class_mapping.json`), and schema definitions in `data/metadata/` are tracked by Git.
- **Manifests**: Sample manifests containing image metadata, deterministic file hashes, and split assignments are stored in `data/metadata/manifests/`.

## Adding a New Dataset Source

1. Subclass `DatasetSource` in `ml/sources/<source_name>.py`.
2. Implement metadata extraction, sample discovery, and source-specific label mapping.
3. Register the source in `data/metadata/dataset_registry.json`.
4. Run validation and manifest generation via `python scripts/build_manifest.py --source <source_name>`.
