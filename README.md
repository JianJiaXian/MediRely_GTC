# MediRely

### Reliable Clinical Context Recovery

> **Retrieve, Don't Hallucinate. Recover, But Verify.**

MediRely helps multimodal medical AI remain reliable when clinical reports are
unavailable. Instead of generating a missing report, it **retrieves real clinical
evidence, recovers useful context, and estimates whether that evidence can be
trusted**.

> ⚕️ **Research prototype — not for clinical use.**
> See [Clinical Disclaimer](#clinical-disclaimer).

---

## Demo

![MediRely demo](assets/medirely_demo.gif)

Given a chest X-ray with no available clinical report, MediRely:

1. encodes the X-ray with **MedSigLIP** for evidence retrieval,
2. searches clinical memory using **NVIDIA cuVS**,
3. retrieves similar real cases,
4. recovers clinical context from the retrieved evidence,
5. estimates **Evidence Reliability**, and
6. compares image-only and recovered-context predictions.

The interface exposes the complete workflow — retrieved evidence, recovered
context, reliability, prediction changes, and the active retrieval backend.

---

## Why MediRely?

Multimodal medical AI often assumes that both medical images and clinical text
are available at inference time. In practice, reports may be **missing, delayed,
inaccessible, or not yet written**.

Two common fallbacks have important limitations:

- **Image-only inference** discards a modality the multimodal model was trained to use.
- **Generating the missing report** can introduce clinical information that is
  not grounded in real evidence.

MediRely takes a different approach:

> **Retrieve real evidence → Recover missing context → Verify its reliability**

MediRely retrieves and aggregates existing clinical evidence. It is **not a
report-generation model**.

---

## Results

### Held-out IU X-Ray Test Set

| Method | Macro AUC |
|---|---:|
| Image only | 0.722 |
| XRV retrieval | 0.777 |
| **MedSigLIP + NVIDIA cuVS retrieval** | **0.819** |

### **0.722 → 0.819 Macro AUC**

with the **MedSigLIP + NVIDIA cuVS GTC retrieval extension**.

A separately evaluated validation-calibrated soft-gated variant reaches
**0.824 macro AUC / 0.015 ECE**.

> These are **GTC extension results** and are separate from the original
> MICCAI study.

See [Detailed Results](#detailed-results) for the full breakdown.

---

## System Architecture

![MediRely architecture](assets/medirely_architecture_v2.gif)

MediRely uses two complementary representations of the same query X-ray.

**MedSigLIP** provides the medical-image embedding used by **NVIDIA cuVS** for
evidence retrieval, while **TorchXRayVision DenseNet-121** provides the image
features used by the validated downstream prediction branch.

Retrieved evidence is aggregated into recovered clinical context, which is
combined with the XRV image features by the existing multimodal classifier.

**Evidence Reliability** is estimated alongside the recovered context and
displayed to the user; it is not directly fed into the base multimodal classifier.

A separately evaluated GTC-only calibrated soft gate can optionally blend the
image-only and recovered-context probabilities.

---

## How It Works

### 1. Encode

The query chest X-ray is encoded by `google/medsiglip-448` into a
**1152-dimensional medical-image embedding** for evidence retrieval.

MedSigLIP is used specifically for the retrieval representation; it does not
replace the image encoder in the validated downstream prediction branch.

### 2. Retrieve

**NVIDIA cuVS** searches the clinical-memory embeddings and returns the Top-K
most similar cases.

The default configuration uses:

```text
Top-K = 10
```

Vectors are L2 normalized and searched using squared Euclidean distance.

For normalized vectors, cosine similarity is recovered as:

```text
cosine_similarity = 1 - distance / 2
```

### 3. Recover

MediRely aggregates evidence from the retrieved cases into a recovered
clinical-context representation.

When source IU X-Ray reports are available locally, the interface can also
display real report excerpts associated with the retrieved cases.

**No missing report text is generated.**

### 4. Verify

Neighbor evidence agreement and retrieval confidence are combined into an
**Evidence Reliability** estimate.

The interface presents this as:

```text
HIGH / MEDIUM / LOW
```

This allows users to inspect whether the recovered context is supported by
consistent retrieved evidence rather than blindly trusting every retrieval.

### 5. Predict

**TorchXRayVision DenseNet-121** provides the image representation used by the
validated downstream prediction branch.

The XRV image features and recovered clinical context are passed to the existing
multimodal classifier to produce the final prediction.

---

## NVIDIA Technology

**NVIDIA cuVS** powers MediRely's GPU clinical-memory retrieval layer.

The GTC implementation uses genuine cuVS brute-force nearest-neighbor search:

```python
from cuvs.neighbors import brute_force
```

cuVS searches over the MedSigLIP clinical-memory embeddings and returns the
Top-K retrieved cases used for context recovery.

When genuine cuVS is active, the application explicitly displays:

```text
NVIDIA cuVS / LIVE
```

The strict WSL2 launcher fails clearly if genuine cuVS cannot be initialized
rather than silently presenting another retrieval backend as cuVS.

> The current IU X-Ray clinical memory contains approximately **5,217 vectors**.
> MediRely does **not claim a cuVS speedup over PyTorch at this scale**.
> cuVS serves as the genuine GPU vector-search backend and provides a path toward
> larger clinical-memory deployments.

---

## Key Features

- 🩻 **Missing-context recovery** for multimodal medical AI
- 🔎 **Real evidence retrieval** instead of generated reports
- 🧠 **MedSigLIP** medical-image embeddings for retrieval
- ⚡ **NVIDIA cuVS** GPU vector search over clinical memory
- 🛡️ **Reliability-aware recovery** using evidence agreement and retrieval confidence
- 🖥️ **Interactive local GPU demo** with transparent backend reporting

---

## Quick Start

### 1. Clone

```bash
git clone https://github.com/JianJiaXian/MediRely_GTC.git
cd MediRely_GTC
```

### 2. Create the Environment

The recommended configuration uses **Windows 11 + WSL2 Ubuntu + NVIDIA GPU**
for genuine NVIDIA cuVS.

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

MediRely uses the gated model:

```text
google/medsiglip-448
```

MedSigLIP weights are **not redistributed by this repository**.

After accepting the applicable HAI-DEF terms, place the model at:

```text
MediRely_GTC/
└── medsiglip-448/
    ├── config.json
    └── ...
```

Alternatively, configure a local model path:

```bash
export MEDIRELY_MEDSIGLIP_DIR=/path/to/medsiglip-448
```

### 4. Public MediRely Artifacts

The sanitized MediRely artifacts required for retrieval, reliability estimation,
and prediction are already included in:

```text
public_artifacts/
```

No additional artifact download is required.

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

Then open:

```text
http://localhost:7860
```

A successful genuine-cuVS launch displays:

```text
NVIDIA cuVS / LIVE
```

### UI-only Preview

To inspect the interface without GPU inference:

```bash
python -m gtc_demo.app.app --preview
```

Preview mode uses recorded, de-identified outputs and **does not claim live
cuVS inference**.

---

# Detailed Results

All results in this section are from the **GTC retrieval extension** evaluated
on the held-out IU X-Ray test set.

## Retrieval Representation Comparison

| Method | Macro AUC |
|---|---:|
| Image only | 0.722 |
| XRV retrieval | 0.777 |
| **MedSigLIP + NVIDIA cuVS retrieval** | **0.819** |

The classifier and evaluation split are held fixed while the retrieval
representation is changed.

The MedSigLIP retrieval representation improves the macro AUC from **0.777**
with the original XRV retrieval representation to **0.819**.

Compared with image-only inference, the complete GTC retrieval extension improves
macro AUC from **0.722 to 0.819**.

---

## GTC-only Calibrated Extension

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

The first example shows coherent retrieved evidence together with a high
reliability estimate.

The second demonstrates less consistent retrieved evidence, resulting in a lower
reliability estimate rather than treating all recovered context as equally
trustworthy.

The external X-ray demonstrates the workflow on a user-provided image. It is
**not part of the benchmark** and should not be interpreted as formal clinical
validation.

---

## Performance

Observed end-to-end demo inference times on the local test machine:

**NVIDIA GeForce RTX 2060 6 GB · WSL2 backend**

| Demo Run | End-to-End Latency |
|---|---:|
| 1 | 670.8 ms |
| 2 | 640.4 ms |
| 3 | 596.7 ms |

These measurements are hardware-specific and represent the observed local demo
runtime.

The first inference after application launch may be slower because of model and
CUDA initialization.

---

# Reproducibility

## Tested Environment

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

## Models and Dependencies

| Component | Role |
|---|---|
| **MedSigLIP** (`google/medsiglip-448`) | Medical-image representation for evidence retrieval |
| **TorchXRayVision DenseNet-121** | Image representation for the validated prediction branch |
| **NVIDIA cuVS** | GPU nearest-neighbor search over clinical memory |
| **Gradio** | Interactive demo interface |

### MedSigLIP

Model:

```text
google/medsiglip-448
```

Retrieval embedding dimension:

```text
1152
```

Official `SiglipImageProcessor` preprocessing is used at **448 × 448**.

### Prediction Encoder

TorchXRayVision:

```text
densenet121-res224-all
```

is used for the validated downstream prediction branch.

### Retrieval

Default retrieval configuration:

```text
Top-K = 10
```

Vectors are L2 normalized and searched using squared Euclidean distance.

For normalized vectors:

```text
cosine_similarity = 1 - distance / 2
```

---

## Public Artifacts

The repository contains four audited researcher-generated artifacts:

| Artifact | Availability | Purpose |
|---|---|---|
| `best_iu_full_image_text_xrv.pth` | `public_artifacts/` | Validated multimodal classifier checkpoint |
| `medsiglip_train_embeddings.npz` | `public_artifacts/` | MedSigLIP retrieval embeddings |
| `iu_train_memory_bank_public.pt` | `public_artifacts/` | Sanitized numerical clinical memory |
| `step4_medsiglip_comparison.json` | `public_artifacts/` | Reliability calibration and evaluation metadata |

The public artifacts were sanitized and validated against the original research
artifacts before release.

They contain **no source X-ray images, full clinical reports, absolute dataset
paths, gated MedSigLIP weights, or authentication tokens**.

### Public-vs-Private Validation

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

Across the validated demo cases:

- Top-K retrieval order: identical
- Similarities: identical within numerical precision
- Reliability: identical
- Prediction: identical
- Maximum cuVS-vs-PyTorch similarity difference: approximately `9.5e-7`

No inference, retrieval, calibration, or evaluation logic was changed during
artifact sanitization.

---

## About Retrieved Report Excerpts

The recorded demo shows retrieved report excerpts because the maintainer's local
research environment has access to the source reports.

Those report texts are **not redistributed in this repository**.

A fresh public installation can reproduce the numerical retrieval, reliability,
and prediction pipeline using the sanitized artifacts.

Displaying the original retrieved report excerpts requires the user to obtain
the source IU X-Ray resources separately.

MediRely never fabricates replacement report text when those excerpts are
unavailable.

---

## IU X-Ray Local Assets

IU X-Ray source images and reports are **not redistributed by this repository**.

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

```text
Top-1 agreement:          100%
Top-K set agreement:      100%
Top-K ordered agreement:  100%
Max similarity delta:     ~9.5e-7
End-to-end parity:        PASS
```

This verifies parity across:

```text
retrieval → reliability → prediction
```

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

For the genuine NVIDIA cuVS configuration, use the WSL2 launcher:

```bash
./run_local_cuvs.sh
```

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

# Research Foundation

The original **MediRely** research established the missing-context recovery
framework and was accepted at the **MICCAI 2026 ML-CDS Workshop**.

**Paper:**  
[MediRely: Reliability-Aware Retrieval for Robust Multimodal Clinical Decision Support](https://openreview.net/forum?id=Ry2H6QuVpc)

The GTC version extends that research into an interactive open-model application
with:

- **MedSigLIP-based medical-image retrieval**
- **NVIDIA cuVS GPU vector search**
- an interactive evidence and reliability interface
- audited public inference artifacts

The **MedSigLIP + NVIDIA cuVS results reported in this repository are GTC
extensions** and are not the original MICCAI paper results.

---

# Citation

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

# Acknowledgments

MediRely builds on:

- **NVIDIA cuVS** for GPU nearest-neighbor retrieval
- **MedSigLIP** (`google/medsiglip-448`) for medical-image retrieval embeddings
- **TorchXRayVision** for the validated DenseNet-121 prediction representation
- **Gradio** for the interactive interface
- **IU Chest X-Ray / NLM Open-i** for the research dataset

The original MediRely research was developed with academic collaborators and
accepted at the **MICCAI 2026 ML-CDS Workshop**.

---

# Clinical Disclaimer

> ⚕️ **Research prototype — not for clinical use.**

MediRely is intended exclusively for research and demonstration.

It is not a medical device, diagnostic system, or production clinical
decision-support tool and must not be used for patient care or real clinical
decision-making.

---

# License

A repository-level code license has not yet been specified.

Third-party models, datasets, and dependencies remain subject to their respective
licenses and terms, including **MedSigLIP / HAI-DEF**, **IU X-Ray / NLM Open-i**,
**NVIDIA cuVS**, **TorchXRayVision**, and other dependencies.

No third-party model weights or source IU X-Ray dataset contents are redistributed
by this repository.
