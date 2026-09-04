# MediRely

**Reliable Clinical Context Recovery**

> **Retrieve, Don't Hallucinate. Recover, But Verify.**

MediRely helps multimodal medical AI remain reliable when clinical reports are
unavailable by **retrieving real clinical evidence, recovering useful context,
and estimating whether that evidence is trustworthy** — instead of hallucinating
a missing report.

> ⚕️ **Research prototype — not for clinical use.** See
> [Clinical Disclaimer](#clinical-disclaimer).

---

## Demo

![MediRely demo](assets/medirely_demo.gif)

The demo shows the complete recovery workflow:

1. A chest X-ray is available while the clinical report is missing.
2. **MedSigLIP** encodes the X-ray for evidence retrieval.
3. **NVIDIA cuVS** searches the clinical memory for similar cases.
4. MediRely recovers clinical context from retrieved evidence.
5. **Evidence Reliability** estimates how trustworthy that recovered context is.
6. The image-only prediction is compared with the MediRely prediction.

The interface exposes the retrieval backend, retrieved evidence, recovered context,
reliability estimate, and downstream prediction in one interactive workflow.

---

## What it does

When a clinical report is unavailable, MediRely:

1. encodes the X-ray with **MedSigLIP** for evidence retrieval,
2. retrieves similar real cases using **NVIDIA cuVS** GPU vector search,
3. aggregates retrieved clinical evidence into recovered context,
4. estimates **evidence reliability**, and
5. uses the recovered context with the existing multimodal prediction branch.

The downstream image representation remains the validated
**TorchXRayVision DenseNet-121** representation.

> **MedSigLIP is used for the evidence-retrieval branch, not as the final
> prediction image encoder.**
>
> MediRely retrieves and aggregates existing clinical evidence. It is **not a
> report-generation model**.

---

## Results

Held-out **IU X-Ray** test set, macro AUC:

| Method | Macro AUC |
|---|---:|
| Image only | 0.722 |
| XRV retrieval | 0.777 |
| **MedSigLIP + NVIDIA cuVS retrieval** | **0.819** |

These results are from the **GTC retrieval extension**.

A separately evaluated GTC-only calibrated soft-gated variant reaches
**0.824 macro AUC / 0.015 ECE**.

See [Detailed Results](#detailed-results) for the full breakdown.

> These numbers are separate from the original MICCAI study. See
> [Research Foundation](#research-foundation).

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/JianJiaXian/MediRely_GTC.git
cd MediRely_GTC
```

### 2. Create the environment

The recommended configuration uses **WSL2 / Ubuntu** with genuine NVIDIA cuVS.

```bash
conda create -n medirely_live python=3.11 -y
conda activate medirely_live

pip install torch==2.8.* torchvision==0.23.* \
  --index-url https://download.pytorch.org/whl/cu129

pip install --extra-index-url https://pypi.nvidia.com \
  cuvs-cu12==24.12.*

pip install -r requirements-wsl-cuvs.txt
```

### 3. Obtain MedSigLIP

MediRely uses:

`google/medsiglip-448`

MedSigLIP is gated and is **not redistributed by this repository**.

Obtain the model from the official Hugging Face repository after accepting the
applicable HAI-DEF terms, then place it at:

```text
MediRely_GTC/
└── medsiglip-448/
    ├── config.json
    └── ...
```

Alternatively:

```bash
export MEDIRELY_MEDSIGLIP_DIR=/path/to/medsiglip-448
```

### 4. Public MediRely artifacts

The sanitized artifacts required for retrieval, reliability estimation, and
prediction are included in:

```text
public_artifacts/
```

They were generated from the validated MediRely research pipeline and audited
before public release.

They contain **no source X-ray images, full clinical reports, absolute dataset
paths, gated MedSigLIP weights, or authentication tokens**.

See:

```text
public_artifacts/ARTIFACTS.md
```

for artifact descriptions, SHA256 checksums, and sanitization details.

### 5. Run

```bash
export MEDIRELY_MODEL_ROOT="$PWD"
./run_local_cuvs.sh
```

Open:

```text
http://localhost:7860
```

When genuine NVIDIA cuVS is active, the interface displays:

```text
NVIDIA cuVS / LIVE
```

---

## UI Preview

The repository also includes recorded, de-identified preview outputs.

No GPU, MedSigLIP model, or live inference is required:

```bash
python -m gtc_demo.app.app --preview
```

Preview mode renders the interface using recorded outputs and **does not claim
live cuVS inference**.

---

## Why MediRely

Many multimodal medical AI systems assume that both the medical image and its
clinical report are available at inference time.

In real workflows, however, the report may be **missing, delayed, inaccessible,
or not yet written**.

Two common fallbacks are problematic:

- **Image-only inference** discards a modality the multimodal model was trained to
  use.
- **Generating a missing report** can introduce hallucinated clinical information
  that was never grounded in real evidence.

MediRely takes a different approach:

> **Retrieve real evidence, recover the missing context, and estimate whether that
> recovered evidence can be trusted.**

---

## Key Features

- **Missing-context recovery** for multimodal medical AI
- **Real evidence retrieval** instead of generated reports
- **MedSigLIP** medical image embeddings for retrieval
- **NVIDIA cuVS** GPU vector search over clinical memory
- **Reliability-aware recovery** using evidence agreement and retrieval confidence
- Existing **TorchXRayVision DenseNet-121** prediction branch preserved
- Interactive local GPU demo
- Transparent live retrieval-backend indicator
- Audited public inference artifacts
- cuVS-vs-PyTorch numerical parity verification

---

## System Architecture

![MediRely architecture](assets/medirely_architecture_v2.gif)

MediRely uses two complementary representations of the same query X-ray:
**MedSigLIP** provides the medical-image embedding used by **NVIDIA cuVS** for
evidence retrieval, while **TorchXRayVision DenseNet-121** provides the image
features used by the validated downstream prediction branch.

Retrieved evidence is aggregated into recovered clinical context, which is combined
with the XRV image features by the existing multimodal classifier.

**Evidence Reliability** is estimated alongside the recovered context and displayed
to the user; it is not directly fed into the base classifier.

**Evidence Reliability is estimated and displayed alongside the prediction.**
It is not directly fed into the base multimodal classifier.

A separately evaluated GTC-only calibrated soft-gate can optionally blend the
image-only and recovered-context probabilities.

---

## How It Works

### 1. Query X-ray

A chest X-ray enters the system while its clinical report is unavailable.

### 2. MedSigLIP retrieval representation

`google/medsiglip-448` encodes the X-ray into a **1152-dimensional medical image
embedding**.

The MedSigLIP representation is used specifically for evidence retrieval.

### 3. NVIDIA cuVS search

NVIDIA cuVS performs GPU nearest-neighbor search over the clinical-memory
embeddings and retrieves the Top-K nearest cases.

Vectors are L2 normalized and cosine similarity is derived from squared Euclidean
distance:

```text
cosine_similarity = 1 - distance / 2
```

### 4. Evidence retrieval

The retrieved cases provide the evidence used for context recovery.

When the user has locally obtained the source IU X-Ray reports, the GUI can also
display real report excerpts for those retrieved cases.

### 5. Context recovery

MediRely aggregates the retrieved evidence into a recovered clinical-context
representation.

**No missing report text is generated.**

### 6. Reliability estimation

Neighbor evidence agreement and retrieval confidence are combined into an
evidence-reliability estimate.

The interface presents this as:

```text
HIGH / MEDIUM / LOW
```

### 7. Prediction

TorchXRayVision DenseNet-121 extracts the image representation used by the
validated prediction branch.

The image representation and recovered clinical context are passed to the
existing multimodal classifier.

---

## NVIDIA Technology

**NVIDIA cuVS** powers MediRely's GPU clinical-memory retrieval layer.

The GTC implementation uses genuine cuVS brute-force nearest-neighbor search:

```python
from cuvs.neighbors import brute_force
```

cuVS searches over the MedSigLIP clinical-memory embeddings and returns the
Top-K nearest cases.

The application explicitly reports the active backend:

```text
NVIDIA cuVS / LIVE
```

If genuine cuVS is requested but unavailable, the strict WSL2 launcher fails
rather than silently presenting another backend as cuVS.

> The current IU X-Ray clinical memory contains approximately 5,217 vectors.
> MediRely does **not claim a cuVS speedup over PyTorch at this scale**.
> cuVS is used as the genuine GPU vector-search backend and provides a path toward
> larger clinical-memory deployments.

---

## Open Models and Dependencies

| Component | Role |
|---|---|
| **MedSigLIP** (`google/medsiglip-448`) | Medical image representation for evidence retrieval |
| **TorchXRayVision DenseNet-121** | Image representation for the validated prediction branch |
| **NVIDIA cuVS** | GPU nearest-neighbor search over clinical memory |
| **Gradio** | Interactive demo interface |

---

# Detailed Results

All results in this section are from the **GTC retrieval extension** evaluated on
the held-out IU X-Ray test set.

### Retrieval representation comparison

| Method | Macro AUC |
|---|---:|
| Image only | 0.722 |
| XRV retrieval | 0.777 |
| **MedSigLIP + NVIDIA cuVS retrieval** | **0.819** |

The classifier and evaluation split are held fixed; the retrieval representation
is changed.

### GTC-only calibrated extension

A validation-calibrated soft gate was additionally evaluated:

| Method | Macro AUC | ECE |
|---|---:|---:|
| MedSigLIP + cuVS, soft-gated | **0.824** | **0.015** |

This calibrated variant is reported separately from the base retrieval result.

> The MedSigLIP/cuVS experiments are GTC extensions and should not be interpreted
> as the original MICCAI paper results.

---

## Demo Examples

| Example | Image-only → MediRely | Evidence Reliability |
|---|---:|---:|
| Strong recovery | No Finding **4% → 96%** | HIGH |
| Ambiguous evidence | Atelectasis **46% → 68%** | MEDIUM |
| External X-ray demo | Cardiomegaly **82% → 98%** | — |

The first example shows coherent retrieved evidence and a high reliability
estimate.

The second demonstrates the opposite behavior: retrieved evidence is less
consistent, so MediRely reports lower reliability rather than blindly trusting
the recovered context.

The external X-ray is included only to demonstrate the workflow on a
user-provided image. It is **not part of the benchmark** and should not be
interpreted as formal clinical validation.

---

## Performance

Observed end-to-end demo inference times on the local test machine:

**NVIDIA GeForce RTX 2060 6 GB · WSL2 backend**

| Demo run | End-to-end latency |
|---|---:|
| 1 | 670.8 ms |
| 2 | 640.4 ms |
| 3 | 596.7 ms |

These are local demo measurements and are hardware-specific.

The first inference after application launch may be slower because of model and
CUDA initialization.

---

# Installation

## Prerequisites

- NVIDIA GPU
- Recent NVIDIA driver with WSL2 GPU support
- Windows 11 + WSL2 Ubuntu recommended for genuine cuVS
- Conda / Miniconda
- Git

### Tested Environment

| Component | Version |
|---|---|
| Host | Windows 11 |
| Backend | WSL2 Ubuntu 24.04 |
| GPU | NVIDIA GeForce RTX 2060 6 GB |
| Python | 3.11 |
| PyTorch | 2.8.0 + cu129 |
| torchvision | 0.23.0 |
| transformers | 4.45.2 |
| NVIDIA cuVS | 24.12 |
| Gradio | 4.44.1 |
| gradio_client | 1.3.0 |
| fastapi | 0.112.2 |
| starlette | 0.38.6 |
| pydantic | 2.9.2 |
| huggingface_hub | 0.25.2 |

---

## Model Setup

MedSigLIP weights are not redistributed.

Obtain `google/medsiglip-448` from its official distribution and either place it
at:

```text
./medsiglip-448/
```

or configure:

```bash
export MEDIRELY_MEDSIGLIP_DIR=/path/to/medsiglip-448
```

Do not commit MedSigLIP weights or authentication tokens.

---

## Public Artifact Setup

The repository contains four audited researcher-generated artifacts:

| Artifact | Availability | Purpose |
|---|---|---|
| `best_iu_full_image_text_xrv.pth` | Included in `public_artifacts/` | validated multimodal classifier checkpoint |
| `medsiglip_train_embeddings.npz` | Included in `public_artifacts/` | MedSigLIP retrieval embeddings |
| `iu_train_memory_bank_public.pt` | Included in `public_artifacts/` | sanitized numerical clinical memory |
| `step4_medsiglip_comparison.json` | Included in `public_artifacts/` | reliability calibration and evaluation metadata |

The public versions were sanitized and validated against the original research
artifacts.

Validation showed:

```text
Private vs public:
retrieval ranking difference = 0
reliability difference       = 0
prediction difference        = 0
```

The sanitized memory bank preserves the numerical information required for
retrieval, reliability estimation, and prediction.

It intentionally excludes:

- full IU X-Ray report text,
- source-image paths,
- patient/study identifiers,
- absolute machine paths.

### About report excerpts

The recorded demo shows retrieved report excerpts because the maintainer's local
research environment has access to the source reports.

Those report texts are **not redistributed in this repository**.

A fresh public installation can reproduce the numerical retrieval, reliability,
and prediction pipeline using the sanitized artifacts. Displaying the original
report excerpts requires the user to obtain the source IU X-Ray resources
separately.

MediRely never fabricates replacement report text when those excerpts are
unavailable.

---

## IU X-Ray Local Assets

IU X-Ray source images and reports are not redistributed by this repository.

After obtaining the dataset from its official source, local assets can be
configured with:

```bash
python scripts/prepare_local_iu_assets.py
```

An example gallery configuration is provided at:

```text
configs/gallery_sources.example.json
```

The real local configuration remains gitignored.

---

## Running the Demo

### Genuine NVIDIA cuVS

```bash
export MEDIRELY_MODEL_ROOT="$PWD"
./run_local_cuvs.sh
```

Then visit:

```text
http://localhost:7860
```

The strict launcher requires genuine cuVS.

If cuVS cannot be initialized, the application fails clearly rather than silently
falling back.

---

## Verifying NVIDIA cuVS

The application startup report checks:

```text
[OK] NVIDIA GPU visible
[OK] MedSigLIP encoder + retrieval bank
[OK] XRV model + classifier
[OK] reliability calibration
[OK] genuine NVIDIA cuVS
```

The GUI then displays:

```text
NVIDIA cuVS / LIVE
```

End-to-end cuVS-vs-PyTorch parity can be tested with:

```bash
python -m gtc_demo.scripts.local_cuvs_parity
```

The validated implementation produced:

- Top-1 agreement: 100%
- Top-K set agreement: 100%
- Top-K ordered agreement: 100%
- maximum similarity difference: approximately `1e-6`
- retrieval → reliability → prediction parity: passed

---

## Windows Fallback

Native Windows uses a PyTorch CUDA exact-search fallback:

```bat
run_local_gpu.bat
```

| Configuration | Retrieval Backend | GUI Badge |
|---|---|---|
| WSL2 | **NVIDIA cuVS** | `NVIDIA cuVS / LIVE` |
| Native Windows | PyTorch CUDA | `PyTorch CUDA / LIVE` |

The fallback is never presented as NVIDIA cuVS.

---

## Repository Structure

```text
MediRely_GTC/
├── assets/
│   ├── medirely_demo.gif
│   └── medirely_architecture_v2.gif
│
├── configs/
│   ├── iu_xray_xrv.yaml
│   └── gallery_sources.example.json
│
├── gtc_demo/
│   ├── app/
│   ├── pipeline/
│   ├── models/
│   ├── retrieval/
│   ├── scripts/
│   ├── demo_assets/
│   └── artifacts/
│
├── models/
├── utils/
│
├── public_artifacts/
│   ├── ARTIFACTS.md
│   ├── best_iu_full_image_text_xrv.pth
│   ├── medsiglip_train_embeddings.npz
│   ├── iu_train_memory_bank_public.pt
│   └── step4_medsiglip_comparison.json
│
├── scripts/
│   ├── download_public_artifacts.py
│   └── prepare_local_iu_assets.py
│
├── QUICKSTART.md
├── README.md
├── requirements-wsl-cuvs.txt
├── requirements-local-gpu.txt
├── run_local_cuvs.sh
└── run_local_gpu.bat
```

Not redistributed:

```text
medsiglip-448/
IU X-Ray source images
IU X-Ray source reports
private full memory bank
real gallery_sources.json
```

---

## Reproducibility

### MedSigLIP

Model:

```text
google/medsiglip-448
```

Model revision:

```text
9cea28a1a1195f665105faa6e8544c112fd960a4
```

Retrieval embedding dimension:

```text
1152
```

Official `SiglipImageProcessor` preprocessing is used at 448×448.

### Prediction Encoder

TorchXRayVision:

```text
densenet121-res224-all
```

is used for the validated prediction branch.

### Retrieval

Default:

```text
Top-K = 10
```

Vectors are L2 normalized and searched using squared Euclidean distance.

Cosine similarity is recovered as:

```text
cosine = 1 - distance / 2
```

### Public Artifact Validation

The sanitized public artifacts were compared directly against the private
research artifacts.

Across all validated demo cases:

- Top-K retrieval order: identical
- Similarities: identical within numerical precision
- Reliability: identical
- Prediction: identical
- maximum cuVS-vs-PyTorch similarity difference: approximately `9.5e-7`

No inference, retrieval, calibration, or evaluation logic was changed during
artifact sanitization.

---

## Research Foundation

The original **MediRely** research established the missing-context recovery
framework and was accepted at the **MICCAI 2026 ML-CDS Workshop**.

**Paper:**  
[MediRely: Reliability-Aware Retrieval for Robust Multimodal Clinical Decision Support](https://openreview.net/forum?id=Ry2H6QuVpc)

The GTC version extends that research into an interactive open-model application
with:

- MedSigLIP-based medical-image retrieval,
- NVIDIA cuVS GPU vector search,
- an interactive evidence/reliability interface,
- audited public inference artifacts.

The **MedSigLIP + NVIDIA cuVS results reported in this repository are GTC
extensions** and are not the original MICCAI paper results.

---

## Clinical Disclaimer

**Research prototype — not for clinical use.**

MediRely is intended exclusively for research and demonstration. It is not a
medical device, diagnostic system, or production clinical decision-support tool
and must not be used for patient care or real clinical decision-making.

---

## Citation

If you use the original MediRely research, please cite:

```bibtex
@inproceedings{
jian2026medirely,
title={MediRely: Reliability-Aware Retrieval for Robust Multimodal Clinical Decision Support},
author={Jia-Xian Jian and Jenq-Neng Hwang and Pau-Choo Chung},
booktitle={ML-CDS 2026: Multimodal Learning and Fusion Across Scales for Clinical Decision Support},
year={2026},
url={https://openreview.net/forum?id=Ry2H6QuVpc}
}
```

The MedSigLIP + NVIDIA cuVS implementation in this repository is a GTC extension
of the research system.

---

## Acknowledgments

MediRely builds on:

- **NVIDIA cuVS** for GPU nearest-neighbor retrieval
- **MedSigLIP** (`google/medsiglip-448`) for medical-image retrieval embeddings
- **TorchXRayVision** for the validated DenseNet-121 prediction representation
- **Gradio** for the interactive interface
- **IU Chest X-Ray / NLM Open-i** for the research dataset

The original MediRely research was developed with academic collaborators and
accepted at the MICCAI 2026 ML-CDS Workshop.

---

## License

A repository-level code license has not yet been specified.

Third-party models, datasets, and dependencies remain subject to their respective
licenses and terms, including **MedSigLIP / HAI-DEF**, **IU X-Ray / NLM Open-i**,
NVIDIA cuVS, TorchXRayVision, and other dependencies.

No third-party model weights or source IU X-Ray dataset contents are redistributed
by this repository.
