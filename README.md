# Cancer Detection in Histopathology Images

An end-to-end machine learning platform and REST API for detecting and classifying cancer in histopathology image patches.

> **Disclaimer:** This project is a research and educational prototype and is **not** intended for clinical diagnosis or medical decision-making.

---

## Current Development Status

- **Phase 1 (Completed):** Backend stabilization, Django REST Framework setup, Docker containerization, CI workflow, and health check API.
- **Phase 2 (Completed):** Multi-Source Histopathology Dataset Pipeline (dataset registry, source adapters, image validation, duplicate detection, leakage-safe splits, preprocessing transforms, CLI workflows).
- **Phase 3 (Upcoming):** Deep Learning Model Architectures (ResNet, EfficientNet, ViT), training routines, and clinical evaluation metrics.

---

## Dataset Pipeline Architecture

```
External Source (PCam / BreaKHis)
              ↓
    Source Adapter Discovery
              ↓
  Image & Integrity Validation
              ↓
    Exact Duplicate Detection
              ↓
     Label Standardization
              ↓
        Dataset Manifest (CSV / JSONL)
              ↓
  Data Leakage-Safe Splitting (Patient / Slide / Stratified)
              ↓
 Preprocessing & Augmentation Configuration
              ↓
   Phase 3 Model-Ready Dataset
```

---

## Supported Dataset Sources

| Source ID | Dataset Name | Task | Target | Default Directory | License |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `pcam` | **PatchCamelyon (PCam)** | Binary Classification | Lymph node metastasis (normal vs tumor) | `data/raw/pcam` | CC0: Public Domain |
| `breakhis` | **BreaKHis** | Binary & Subtype Classification | Breast tumor malignancy (benign vs malignant) | `data/raw/breakhis` | Academic Free License |

---

## Dataset Acquisition & Workflow

### 1. View Acquisition Instructions
```bash
python scripts/download_dataset.py --source pcam
python scripts/download_dataset.py --source breakhis
```

### 2. Validate Dataset Integrity & Check for Duplicates
```bash
python scripts/validate_dataset.py --source pcam
```

### 3. Generate Manifest and Create Leakage-Safe Splits
```bash
python scripts/build_manifest.py \
    --source pcam \
    --create-splits \
    --train-ratio 0.70 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --seed 42 \
    --format both
```

---

## Data Leakage Prevention

To ensure clinical and statistical validity, the splitting pipeline employs a strict hierarchy:
1. **Patient-Level Grouping:** When patient identifiers are available (e.g., BreaKHis), all image patches from a patient are strictly assigned to a single split.
2. **Slide-Level Grouping:** When only slide identifiers are available, all patches from a slide remain in one split.
3. **Duplicate Hash Isolation:** Exact duplicate images (identical SHA-256 hashes) are detected and restricted from crossing split boundaries.
4. **Stratified Splitting:** If no grouping identifiers exist (e.g., PCam), deterministic stratified splitting preserves class balance across splits.

---

## Storage & Git Tracking Policy

- `data/raw/`: Raw, unmodified images (Git-ignored).
- `data/interim/`: Validated intermediate data (Git-ignored).
- `data/processed/`: Final split datasets (Git-ignored).
- `data/metadata/`: Dataset registry (`dataset_registry.json`), class mappings (`class_mapping.json`), and manifests (`manifests/`) are tracked by Git.

---

## Running Backend & Tests

```bash
# Run complete test suite (45 tests)
./venv/bin/pytest -v

# Run Django system checks
./venv/bin/python backend/manage.py check --settings=config.settings.development
```
