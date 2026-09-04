# MediRely — Public Artifacts Manifest

These are the **researcher-generated, audited-safe** artifacts that ship with the
public repository so the demo and its numbers are reproducible **without**
redistributing any gated model weights, source images, or clinical report text.

Every file below was produced by a sanitization pass that:
- **preserves** all numerical values used at inference (weights, embeddings,
  labels, metrics, row order) — verified bit/numerically identical to the private
  originals (max embedding diff `0.00e+00`, checkpoint tensors `torch.equal`), and
- **removes** absolute filesystem paths, source filenames, study/patient IDs, and
  full clinical report text.

None of the audited files contained tokens, API keys, `.env` contents, or
Windows/`/home/<user>` paths after sanitization (privacy scan: clean).

| File | Purpose | Source | Researcher-generated | Size | SHA256 | Loader path | Required? | Sanitization | License |
|---|---|---|---|---|---|---|---|---|---|
| `best_iu_full_image_text_xrv.pth` | Trained multimodal classifier weights (image+text head over TorchXRayVision DenseNet-121) + BoW `text_vocab` | Our training run on IU X-Ray | Yes | 38 MB | `93c4761b830d5e5323ef812fab0fadaf0cc9a3689d2a46ad59299619a097f3ce` | `gtc_demo.pipeline.medirely_gtc` (checkpoint) | Yes (live demo) | Dropped `epoch`,`val_metrics`; kept `model_state`,`text_vocab`,`findings`,`config_model`,`model_type`; **weights unchanged** (`torch.equal`=True) | Repo license (researcher-generated) |
| `medsiglip_train_embeddings.npz` | MedSigLIP (1152-d) embeddings of the 5217 training images, for the cuVS retrieval bank | Encoded by us with MedSigLIP | Yes (embeddings; weights not included) | 24 MB | `5f2c57b622b7f842b29a760fad51b6c071835da6367581c36c335045cdbc924d` | `gtc_demo.pipeline.medirely_gtc` (`ms_emb`) | Yes (config C / cuVS) | Removed `image_paths`,`study_ids`; kept `embeddings`(5217,1152 f32),`labels`,`row_ids`; **values identical** (max Δ `0.00e+00`), row order preserved | Repo license; derived from gated MedSigLIP (weights NOT distributed) |
| `iu_train_memory_bank_public.pt` | Sanitized memory bank: XRV image+text embeddings (256-d) + weak labels for the validated retriever | Encoded by us | Yes | 11 MB | `e12d02ab82903f5dc4d7423c1e99cd302f5029984e02cc49c4c4c9c99ef279b6` | `models.pseudo_modality_bridge.MemoryBank.load` | Optional (public alt. to private bank) | `report_texts` and `image_paths` emptied (`['']*N`); embeddings/labels **identical** (`torch.equal`=True) | Repo license (embeddings only; no source reports/images) |
| `step4_medsiglip_comparison.json` | Evaluation metrics + reliability-gate calibration (macro AUC 0.722/0.777/0.819, ECE, gate coefficients) | Our evaluation | Yes | 3 KB | `fea744270095fed8a8f304e06d5a1a7c30858c9c7ccb4f0b550695456abac295` | `gtc_demo.app.adapters.LogisticGate.from_artifact` | Yes (reliability gate) | No paths/IDs present; **metrics/calibration unchanged** (byte-identical to source) | Repo license (researcher-generated) |

## What is NOT here (obtain from the official source)

| Not distributed | Why | Where to get it |
|---|---|---|
| `medsiglip-448/` weights | Gated HAI-DEF model, not ours to redistribute | https://huggingface.co/google/medsiglip-448 (accept the license) |
| IU Chest X-Ray images (`bank_images/`, `demo_images/`, PNG/DICOM) | Licensed third-party source imagery | https://openi.nlm.nih.gov/ |
| Full clinical report text | Source clinical text; not redistributed | Comes with the IU X-Ray source above |

## The two memory-bank modes

The pipeline supports **both** banks with identical numerical outputs:

- **Private (full) bank** `outputs/memory_banks/iu_train_memory_bank_xrv.pt` — includes
  real report excerpts, so the live GUI shows retrieved report snippets. Kept local /
  gitignored. Reports are **not** fabricated for the public bank.
- **Public (sanitized) bank** `iu_train_memory_bank_public.pt` (this folder) — same
  embeddings/labels; report excerpts render blank. Retrieval, reliability, and
  predictions are **unchanged** because they depend on the numerical text embeddings,
  not the report strings.

The MedSigLIP npz loader auto-detects which bank you have: with `image_paths` present
it asserts exact path-order alignment; without them it verifies count + row-id
checksum. Row `i` always corresponds to bank row `i`.

## Verify

```bash
cd public_artifacts && sha256sum -c <<'SUMS'
93c4761b830d5e5323ef812fab0fadaf0cc9a3689d2a46ad59299619a097f3ce  best_iu_full_image_text_xrv.pth
5f2c57b622b7f842b29a760fad51b6c071835da6367581c36c335045cdbc924d  medsiglip_train_embeddings.npz
e12d02ab82903f5dc4d7423c1e99cd302f5029984e02cc49c4c4c9c99ef279b6  iu_train_memory_bank_public.pt
fea744270095fed8a8f304e06d5a1a7c30858c9c7ccb4f0b550695456abac295  step4_medsiglip_comparison.json
SUMS
```

> ⚕️ Research prototype — **not for clinical use.**
