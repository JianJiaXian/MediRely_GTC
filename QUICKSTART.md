# MediRely — Quick Start

**Reliable Clinical Context Recovery** · *Retrieve, Don't Hallucinate. Recover, But Verify.*

When a clinical report is missing, MediRely retrieves **real** similar cases with
**MedSigLIP + NVIDIA cuVS**, recovers clinical context from their **real reports**,
and estimates whether that evidence can be trusted — then supports the existing
multimodal prediction (image encoder stays TorchXRayVision DenseNet‑121).

| Method (held-out IU X-Ray, macro AUC) | Score |
|---|---:|
| Image only | 0.722 |
| XRV retrieval | 0.777 |
| **MedSigLIP + NVIDIA cuVS** | **0.819** |

---

## 1. Instant UI preview (no GPU, no model)

```bash
conda create -n medirely_live python=3.11 -y && conda activate medirely_live
pip install -r requirements-wsl-cuvs.txt
python -m gtc_demo.app.app --preview     # → http://127.0.0.1:7860
```
Renders the interface from **recorded** verified outputs for the 3 demo cases.
(No live inference; never claims live cuVS.)

## 2. Live demo with genuine NVIDIA cuVS (WSL2 + NVIDIA GPU)

```bash
# CUDA-12 PyTorch + genuine cuVS (Linux/WSL2 wheels)
pip install torch==2.8.* torchvision==0.23.* --index-url https://download.pytorch.org/whl/cu129
pip install --extra-index-url https://pypi.nvidia.com cuvs-cu12==24.12.*
pip install -r requirements-wsl-cuvs.txt

# place ./medsiglip-448/ (gated on Hugging Face) and ./artifacts_local/ (see README)
export MEDIRELY_MODEL_ROOT="$PWD"
./run_local_cuvs.sh                        # header shows "NVIDIA cuVS / LIVE"
```
Open **http://localhost:7860** → pick a demo case → **Recover with MediRely**.

## 3. Windows-native fallback (PyTorch CUDA, no cuVS)

```bat
run_local_gpu.bat                          # header shows "PyTorch CUDA / LIVE"
```

---

Full architecture, installation, artifact setup, reproducibility, and disclaimers:
see **[README.md](README.md)**.

> ⚕️ Research prototype — **not for clinical use.**
