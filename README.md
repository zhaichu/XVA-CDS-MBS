
---

## 2 — `xva‑cds‑mbs‑engine` / README.md

```markdown
# XVA Engine for CDS + MBS Portfolios 📈

Python framework that **simulates joint rate–credit dynamics, computes EE, CVA, FVA, PFE, and runs GAN‑based COVID stress tests** for mixed CDS/MBS books.

---

## Highlights
- **Hull‑White 2‑Factor Rates**: PCA‑derived factors, AR(1) mean reversion, analytic zero‑curve fitting.
- **Credit Spread Paths**: Correlated with rates via Gaussian copula or user‑defined β.
- **PyTorch GAN Stressor**: Trains on 2020 market data to synthesize extreme rate‑spread scenarios.
- **Full Exposure Stack**: EE term structure, Monte‑Carlo VaR/CVaR, 97.5 % PFE, incremental XVA charts.

---

## Quick Demo

```bash
# set up
conda env create -f environment.yml
conda activate xva

# run 10k path simulation for a sample CDS + MBS position set
python XVA.py \
       --date  2021-09-10 \
       --years 6          \
       --cds   data/cds.csv \
       --paths 10000 --stress 2000 --epochs 3000
