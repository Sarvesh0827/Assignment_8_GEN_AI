# Mini Diffusion Model (DDPM → DDIM)

Assignment notebook: train a small DDPM on MNIST, then sample with DDPM (100 steps) and DDIM (50 / 20 steps).

## What’s in this repo

| Path | Contents |
|---|---|
| `Mini_Diffusion_DDPM_to_DDIM.ipynb` | Full assignment (code + discussion) |
| `figures/` | Forward diffusion, training loss, DDPM vs DDIM samples |
| `checkpoints/mini_ddpm_mnist.pt` | Trained weights (skip retraining if present) |
| `requirements.txt` | Python dependencies |

## Setup (Mac)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook Mini_Diffusion_DDPM_to_DDIM.ipynb
```

Then **Run All**. If `checkpoints/mini_ddpm_mnist.pt` is present, training is skipped and the notebook loads the saved model. MNIST downloads automatically into `data/` on first run.

Apple Silicon uses MPS automatically when CUDA is unavailable (`torch.backends.mps`).

## Deliverables in the notebook

- (a) Forward diffusion visualization
- (b) Time-conditioned CNN + training loss
- (c) DDPM sampling (100 steps)
- (d) DDIM sampling (50 and 20 steps) and comparison
