#!/usr/bin/env python3
"""
xva_cds_mbs.py
Monte‑Carlo CVA / FVA with 2‑Factor Hull‑White + GAN Stress Testing
Author: <Your Name>   Date: 2025‑07‑03
"""
from __future__ import annotations

import argparse, math, pathlib, logging, sys
from datetime import datetime, timedelta
from pickle import FALSE
from typing import Callable, Tuple, Dict, List

import numpy as np
import pandas as pd
from scipy import interpolate, stats
from pandas_datareader.data import DataReader
from sklearn.decomposition import PCA          # for HW calibration
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import date, timedelta
from scipy.interpolate import PchipInterpolator

# ──────────────────── Globals ───────────────────────────────────────────
DATA = pathlib.Path("data"); DATA.mkdir(exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# ──────────────────── CLI ───────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    P = argparse.ArgumentParser(description="XVA Monte‑Carlo engine with GAN stress")
    P.add_argument("--cds",    default="/Users/zhaimenghe/PycharmProjects/dynamic_hedging/data/cds.csv", help="Kaggle CDS data (PX5)")
    P.add_argument("--years",  type=int, default=1, help="Simulation horizon (years)")
    P.add_argument("--paths",  type=int, default=10_000, help="MC paths")
    P.add_argument("--stress", type=int, default=2_000, help="GAN stress paths")
    P.add_argument("--epochs", type=int, default=3_000, help="GAN training epochs")
    P.add_argument("--rec",    type=float, default=0.40, help="Recovery rate")
    P.add_argument("--fund",   type=float, default=0.002, help="Funding spread (dec)")
    P.add_argument("--curve", help="CSV with maturity_years,zero_rate(%)")
    P.add_argument("--date", default=None,
                   help="Valuation date yyyy-mm-dd (used if --curve omitted)")
    return P.parse_args()

# ─────────── Yield curve & Hull‑White 2F ────────────────────────────────
def bootstrap_zero_curve(val_date: str | date,
                          out_csv: str | None = None
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pulls SOFR + DGS1/2/3/5/7/10 from FRED on `val_date`,
    builds a first‑cut zero curve (continuous compounding),
    returns (maturities, zero_rates) both as np.ndarray.
    Optionally saves a CSV compatible with `load_yield_curve()`.
    """
    logging.debug("→ bootstrap_zero_curve called for %s", val_date)
    if isinstance(val_date, str):
        val_date = pd.to_datetime(val_date).date()

    start = val_date - timedelta(days=4000)

    # --- overnight node ---
    sofr = DataReader("SOFR", "fred", start, val_date).iloc[-1, 0] / 100
    tau_on = 1.0 / 365.0                # 一天≈0.00274y
    dfs = {0.0: 1.0, tau_on: 1 / (1 + sofr * tau_on)}

    # --- 1Y…10Y coupon‑bearing yields (近似零息) ---
    tenors = np.array([1, 2, 3, 5, 7, 10], dtype=float)
    for t in tenors.astype(int):
        y = DataReader(f"DGS{t}", "fred", start, val_date).iloc[-1, 0] / 100
        dfs[t] = math.exp(-y * t)

    # --- convert DF → continuous zero rate ---
    mat = np.array(sorted(dfs))
    dfv = np.array([dfs[m] for m in mat])
    zr  = -np.log(dfv) / np.where(mat == 0, 1, mat)   # 避免 0/0
    zr[mat == 0] = sofr                               # 0Y 点

    if out_csv:
        pd.DataFrame({"maturity_years": mat,
                      "zero_rate": zr * 100}).to_csv(out_csv, index=False)

    return mat, zr

def load_yield_curve(csv: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv)
    if not {"maturity_years","zero_rate"}.issubset(df.columns):
        raise ValueError("CSV must contain maturity_years, zero_rate columns")
    return df["maturity_years"].values, df["zero_rate"].values / 100.0

def phi_from_yield(mat: np.ndarray, zr: np.ndarray) -> Callable[[float], float]:
    f = interpolate.CubicSpline(mat, zr, bc_type="natural")
    return lambda t: float(f(t))

def calibrate_hw2f_monthly(mat, zr, val_date, lookback_years=10):

    end   = pd.to_datetime(val_date)
    start = end - pd.DateOffset(years=lookback_years)
    tick  = {"GS1":"y1", "GS5":"y5", "GS10":"y10"}

    raw = pd.concat([DataReader(t,"fred",start,end).rename(columns={t:n})
                     for t,n in tick.items()], axis=1).dropna() / 100
    df  = raw.resample("ME").last().dropna()    # 月末

    lvl = df.subtract(df.mean())
    pcs = PCA(n_components=2).fit_transform(lvl)

    dt  = 1/12
    def ou(x):
        rho = np.corrcoef(x[:-1], x[1:])[0,1]
        a   = max(-np.log(max(rho,1e-6)) / dt, 0.01)
        sig = np.std(x[1:] - rho*x[:-1]) / np.sqrt(dt)
        return a, sig

    a1,s1 = ou(pcs[:,0]); a2,s2 = ou(pcs[:,1])
    a1,a2 = np.clip([a1,a2], 0.05, 3.0)
    print(f"a1={a1:.3f}, a2={a2:.3f}, s1={s1:.4f}, s2={s2:.4f}")
    phi   = phi_from_yield(mat, zr)
    return dict(a1=a1,a2=a2,s1=s1,s2=s2,phi=phi)


def sim_hw2f(par: Dict[str,float|Callable], T: float, dt: float,
             N: int, seed: int = 42) -> np.ndarray:
    """Return shape (N, steps) of short‑rate paths."""
    np.random.seed(seed)
    a1,a2,s1,s2,phi = (par[k] for k in ("a1","a2","s1","s2","phi"))
    steps = int(T/dt); rho = 0.5
    x1 = np.zeros(N, dtype=float)
    x2 = np.zeros(N, dtype=float)
    rates = np.zeros((N,steps), dtype=float)

    for t in range(steps):
        z1 = np.random.normal(size=N)
        z2 = rho*z1 + math.sqrt(1-rho**2)*np.random.normal(size=N)
        x1 += (-a1*x1)*dt + s1*math.sqrt(dt)*z1
        x2 += (-a2*x2)*dt + s2*math.sqrt(dt)*z2
        rates[:,t] = x1 + x2 + phi((t+1)*dt)
    return rates

# ────────────── CDS & macro factors ─────────────────────────────────────
def load_cds(csv: str) -> pd.Series:
    df = pd.read_csv(csv, usecols=["Date","PX5"]).dropna()
    df["Date"] = pd.to_datetime(df["Date"])
    # PX5 以 bp 表示，除以 10000 → 百分比 (dec.)
    return (df.groupby("Date")["PX5"].mean() / 10000.0).rename("credit")


def build_factors(credit: pd.Series) -> pd.DataFrame:
    idx = credit.index                       # Full date range from CDS

    # ----- FRED series (fetch once) -------
    s, e = idx.min(), idx.max()
    sofr  = DataReader("SOFR",        "fred", s, e) / 100
    fedf  = DataReader("FEDFUNDS",    "fred", s, e) / 100
    m30   = DataReader("MORTGAGE30US","fred", s, e) / 100
    ust10 = DataReader("GS10",        "fred", s, e) / 100

    # SOFR 缺 2015‑2018，用 FEDFUNDS 兜底再 F‑fill
    sofr_full = sofr.reindex(idx).fillna(fedf).fillna(method="ffill")

    # 其余序列直接 reindex → F‑fill
    m30_full  = m30.reindex(idx).fillna(method="ffill")
    ust10_full= ust10.reindex(idx).fillna(method="ffill")

    df = pd.DataFrame({
        "credit": credit,
        "sofr":   sofr_full.squeeze(),
        "mort30": m30_full.squeeze(),
        "ust10":  ust10_full.squeeze()
    })

    df["mbs"] = df["mort30"] - df["ust10"]
    return df


# ──────────────── MBS pricing ───────────────────────────────────────────
def psa(month: int) -> float:
    cpr = min(0.06 * month / 30.0, 0.06)
    return 1.0 - (1.0 - cpr)**(1/12)

def mbs_cf(bal: float, coupon: float, terms: int = 360) -> np.ndarray:
    r = coupon
    pmt = bal * r/12 / (1 - (1+r/12)**-terms)
    b   = bal
    cf: List[float] = []
    for m in range(1, terms+1):
        intr = b * r / 12
        prin = pmt - intr
        pre  = (b - prin) * psa(m)
        cf.append(intr + prin + pre)
        b -= prin + pre
    return np.array(cf, dtype=float)

def mbs_price(bal: float, coupon: float,
              disc_daily: np.ndarray,
              oas_scalar: float) -> float:
    cf   = mbs_cf(bal, coupon)                   # 360 元素
    idx  = np.minimum((np.arange(1,len(cf)+1)*21 - 1),
                      len(disc_daily)-1)
    disc_m = disc_daily[idx.astype(int)]
    adj  = disc_m * np.exp(-oas_scalar * np.arange(1,len(cf)+1)/12)
    return float((cf * adj).sum())

# ───────────── Exposure / CVA / FVA ─────────────────────────────────────
def exposure_paths(rates: np.ndarray,
                   credit: pd.Series,           # 日频 CDS spread (decimal)
                   mbs_oas: pd.Series,          # 日频 MBS OAS (decimal) – 先不用可置0
                   notional_cds: float,
                   bal_mbs: float,
                   coupon_mbs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (net_exposure[N,steps], discount[N,steps])
    """
    dt     = 1/252
    steps  = rates.shape[1]
    disc   = np.exp(-np.cumsum(rates, axis=1)*dt)

    # --- CDS MtM --------------------------------------------------------
    dv01   = 4.0 * notional_cds * 1e-4          # 4y DV01 约 4bp/Notional
    cr_raw = credit.values
    # 如 CDS 历史不足，向后填充最后一个值
    if len(cr_raw) < steps:
        cr = np.pad(cr_raw, (0, steps-len(cr_raw)), 'edge')
    else:
        cr = cr_raw[:steps]
    cds_mtm = dv01 * (cr - cr[0])
    # shape (steps,)
    # --- MBS MtM --------------------------------------------------------
    mbs_mtm = np.zeros_like(rates)              # shape (N,steps)
    base    = mbs_price(bal_mbs, coupon_mbs, disc[0], 0.0)
    for p in range(rates.shape[0]):
        mbs_mtm[p] = (mbs_price(bal_mbs, coupon_mbs, disc[p], 0.0) - base)

    return cds_mtm + mbs_mtm, disc

def cva_fva(net: np.ndarray, disc: np.ndarray, lgd: float,
            funding_spread: float, lam: float = 0.02) -> Tuple[float,float]:
    epe = np.mean(np.maximum(net, 0.0), axis=0)
    ene = np.mean(np.maximum(-net,0.0), axis=0)
    dt  = 1/252
    df  = np.mean(disc, axis=0)
    surv = np.exp(-lam * np.arange(len(epe)) * dt)
    dPD  = surv[:-1] - surv[1:]
    cva = float((df[1:] * epe[1:] * dPD * lgd).sum())
    fva = float((df * (ene - epe) * funding_spread * dt).sum())
    return cva, fva

# ─────────────── PyTorch GAN ────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self, latent: int, seq_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,seq_len), nn.Sigmoid()
        )
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, seq_len: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len,128), nn.LeakyReLU(0.2),
            nn.Linear(128,128), nn.LeakyReLU(0.2),
            nn.Linear(128,1), nn.Sigmoid()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def prep_seq(series: np.ndarray, win: int) -> Tuple[np.ndarray, float]:
    """Create overlapping windows and return (samples, scaler)"""
    if len(series) < win + 1:  # 至少要有 win+1 才能滑窗
        win = len(series) - 1
    arr = np.array([series[i:i + win] for i in range(len(series) - win)])
    scaler = arr.max()
    return arr / scaler, scaler

def train_gan_pt(real: np.ndarray, latent: int, epochs: int = 3_000,
                 batch: int = 64, log: int = 500) -> Generator:
    seq_len = real.shape[1]
    G, D = Generator(latent,seq_len).to(DEVICE), Discriminator(seq_len).to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=3e-4)
    opt_D = optim.Adam(D.parameters(), lr=3e-4)
    bce   = nn.BCELoss()
    real  = torch.tensor(real, dtype=torch.float32, device=DEVICE)
    half  = batch // 2

    for e in range(epochs+1):
        # --- Discriminator step ---
        idx = torch.randint(0, real.size(0), (half,))
        real_b = real[idx]
        z = torch.randn(half, latent, device=DEVICE)
        fake_b = G(z).detach()
        loss_D = (bce(D(real_b), torch.ones_like(real_b[:,:1])) +
                  bce(D(fake_b), torch.zeros_like(fake_b[:,:1])))
        opt_D.zero_grad(); loss_D.backward(); opt_D.step()

        # --- Generator step ---
        z = torch.randn(batch, latent, device=DEVICE)
        gen = G(z)
        loss_G = bce(D(gen), torch.ones_like(gen[:,:1]))
        opt_G.zero_grad(); loss_G.backward(); opt_G.step()

        if log and e % log == 0:
            logging.info("Epoch %4d  D=%.3f  G=%.3f", e, loss_D.item(), loss_G.item())
    return G

#──────────────────── MAIN PIPELINE ──────────────────────────────────────
def main():
    args = parse_args()# 直接返回数组
    print("DEBUG args.years =", args.years)
    # ── 1. Zero‑curve 来源 ──────────────────────────────
    if args.curve:  # 用户自带 CSV
        print("loaded yield curve")
        mat, zr = load_yield_curve(args.curve)
    else:  # 自动 FRED bootstrap
        val_dt = args.date or date.today().strftime("%Y-%m-%d")
        logging.info("No --curve supplied, auto‑bootstrapping FRED curve for %s",
                     val_dt)
        print("bootstrapping: " , val_dt)
        mat, zr = bootstrap_zero_curve(val_dt)

    hw_par = calibrate_hw2f_monthly(mat, zr, val_date=args.date, lookback_years=10)
    # 2. Factors
    credit  = load_cds(args.cds)
    factors = build_factors(credit)

    # 3. Monte‑Carlo baseline
    dt, steps = 1/252, int(args.years/ (1/252))
    rates = sim_hw2f(hw_par, args.years, dt, args.paths)

    net, disc = exposure_paths(
        rates,
        factors["credit"],  # credit pd.Series
        factors["mbs"],  # mbs_oas pd.Series (暂未用)
        notional_cds=5e8,
        bal_mbs=5e6,
        coupon_mbs=0.045
    )
    cva_mc, fva_mc = cva_fva(net, disc, 1 - args.rec, args.fund)

    logging.info("Baseline CVA_MC=%.0f  FVA=%.0f", cva_mc, fva_mc)

    # 4. GAN stress credit‑spread
    # ==== Stress paths from HISTORICAL TAIL (auto‑scale) ==========
    win = 250
    hist = factors["credit"].values

    tail_windows = [hist[i:i + win] for i in range(len(hist) - win)
                    if hist[i:i + win].max() > np.quantile(hist, 0.95)]
    np.random.shuffle(tail_windows)
    stress_raw = np.array(tail_windows[:args.stress])

    hist_max = np.max([w.max() for w in tail_windows])

    def build_credit_paths(mult: float) -> np.ndarray:
        g_scaled = mult * stress_raw  # 放大尾部窗口
        steps = rates.shape[1]
        full = np.pad(g_scaled, ((0, 0), (0, steps - win)), mode="edge")
        return full

    # --- first guess multiplier so that median peak ≈ 1.1 * hist_max
    mult0 = 1.1 * hist_max / stress_raw.max()
    credit_paths = build_credit_paths(mult0)

    # -- KS / Cover check -------------------------------------------------
    gen_tail = credit_paths.max(axis=1)
    hist_tail = factors["credit"][factors["credit"] >
                                  np.quantile(factors["credit"], 0.95)].values
    ks = stats.ks_2samp(hist_tail, gen_tail)
    cover = np.mean((gen_tail > 0.8 * hist_max) & (gen_tail < 1.2 * hist_max))

    # ---- if KS 还是 >0.5, 逐步减小 mult 直到通过 -----------------------
    while ks.statistic > 0.5 and mult0 > 0.2:
        mult0 *= 0.8
        credit_paths = build_credit_paths(mult0)
        gen_tail = credit_paths.max(axis=1)
        ks = stats.ks_2samp(hist_tail, gen_tail)
        cover = np.mean((gen_tail > 0.8 * hist_max) & (gen_tail < 1.2 * hist_max))

    logging.info("Auto‑scale: mult=%.2f  KS=%.3f  PeakCover=%.1f%%",
                 mult0, ks.statistic, cover * 100)

    # ---- Stress CVA loop -----------------------------------
    stress_val = []
    for g_full in tqdm(credit_paths, desc="Stress paths"):
        cr = pd.Series(g_full, index=factors.index[:steps])

        net_s, disc_s = exposure_paths(
            rates[:1], cr, factors["mbs"],
            notional_cds=5e8,  # 与 Baseline 保持同名义
            bal_mbs=5e6,
            coupon_mbs=0.045
        )
        cv, _ = cva_fva(net_s, disc_s, 1 - args.rec, args.fund)
        stress_val.append(cv)

    stress = np.array(stress_val)
    mean = stress.mean()
    tail = stress[stress > np.quantile(stress, 0.95)].mean()
    pfe99 = np.quantile(stress, 0.99)

    logging.info("Stress CVA: mean=%.0f  CVaR95=%.0f  PFE99=%.0f",
                 mean, tail, pfe99)
    if pfe99 > 2 * cva_mc:
        logging.info("✓ PFE99 > 2 × Baseline  ⇒  尾部放大充分⇒  模型稳健")
# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.exception("Fatal error: %s", e)
        sys.exit(1)
