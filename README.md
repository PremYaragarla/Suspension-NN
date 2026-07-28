# Suspension Displacement Prediction — GRU

> A recurrent neural network that predicts vehicle body and tire displacement directly from a road profile, learned from Quanser active-suspension bench data with no physical parameters supplied.

![Class E predictions vs ground truth](figures/classE_predictions.png)

---

## Overview

**The problem:** Predicting how a suspension responds to the road is a core vehicle-dynamics task, and the standard tool is the quarter-car model. It is cheap and interpretable, but it requires accurate mass/stiffness/damping values and it assumes all behaviour is linear. Real suspensions have bushings, friction, and travel limits that introduce nonlinearity a linear model cannot represent. The objective of this project is to predict body and tire displacement simultaneously from road elevation alone, without identifying any physical parameters.

**The approach:** A GRU-based recurrent network maps a sliding 1,000-sample window of ISO 8608 road elevation (1 second of road history at 1 kHz) to the body and tire displacement at the end of that window. Two stacked GRU layers compress the road history into a hidden state, and a three-layer MLP decodes it into the two displacements. I chose a GRU over a plain RNN because suspension response depends on road history over hundreds of milliseconds, which is exactly the range where a vanilla RNN's gradients vanish. Training uses all five roughness classes at once, with global z-score normalisation — MinMax scaling failed because Class E amplitudes are roughly an order of magnitude larger than Class A, so a shared MinMax range squashed the smooth roads to nearly nothing and gave poor accuracy on smoother roads.

**Outcome:** Correlation with ground truth is >= 0.997 on every road class, and body RMSE stays between 0.020 mm (Class A) and 0.071 mm (Class E) against displacements measured in millimetres. More telling than the error numbers: the model behaves sensibly on inputs it never saw in training — speed bumps, potholes, and sine sweeps all produce physically plausible responses, and a frequency sweep recovers the suspension's two natural resonances even though it was only ever trained on broadband road data.

---

## Results

Evaluated on the held-out period of each ISO 8608 class, denormalised back to metres before scoring. All figures and metrics below come from the checkpoint `gru_bs128_s5_seq1000_100k.pt` — batch size 128, stride 5, a 1,000-sample window, trained on the full 100,000 samples per class.

**Body displacement**

| Class | RMSE [mm] | MAE [mm] | NRMSE / range | NRMSE / std | Correlation |
|---|---|---|---|---|---|
| A | 0.0204 | 0.0173 | 1.81 % | 13.28 % | 0.9970 |
| B | 0.0234 | 0.0194 | 0.88 % | 6.14 % | 0.9993 |
| C | 0.0293 | 0.0230 | 0.53 % | 3.91 % | 0.9996 |
| D | 0.0482 | 0.0380 | 0.36 % | 2.84 % | 0.9998 |
| E | 0.0709 | 0.0555 | 0.24 % | 1.99 % | 0.9999 |

**tire displacement**

| Class | RMSE [mm] | MAE [mm] | NRMSE / range | NRMSE / std | Correlation |
|---|---|---|---|---|---|
| A | 0.0142 | 0.0112 | 0.91 % | 6.62 % | 0.9983 |
| B | 0.0207 | 0.0162 | 0.60 % | 4.36 % | 0.9995 |
| C | 0.0360 | 0.0282 | 0.56 % | 4.16 % | 0.9996 |
| D | 0.0623 | 0.0494 | 0.50 % | 3.65 % | 0.9997 |
| E | 0.1002 | 0.0786 | 0.39 % | 2.98 % | 0.9998 |

RMSE and MAE are absolute errors in millimetres; NRMSE normalises them by the signal's range and by its standard deviation, so those columns are comparable across classes in a way the raw errors are not.

Absolute error grows with roughness, but *relative* accuracy moves the other way — and the NRMSE/std column makes it quantitative. Body error normalised by signal spread is 13.3 % on Class A against 1.99 % on Class E, about a sevenfold gap, even though Class A has the smallest RMSE in millimetres. The reason is that Class E's much larger displacements dominate the MSE during training, so the loss has little incentive to resolve Class A's small-amplitude detail. This shows up as a slight negative bias on the smoothest profiles and is the clearest target for the next round of work. The tire channel is tighter across the board (6.6 % down to 3.0 % NRMSE/std) because tire displacement tracks the road more directly and has less of the delayed, filtered dynamics that make the body harder to predict.

![Training and validation loss](figures/training_loss.png)

Training was stopped when the validation loss spiked rather than letting the early-stopping counter run out; the best checkpoint by validation MSE is the one kept.

![Predicted vs actual scatter](figures/classE_correlation.png)

The parity plot shows the errors are unbiased and evenly spread rather than concentrated at the extremes of travel — the model is not simply clipping large excursions.

### A learned transfer function

The strongest evidence that the network learned suspension *dynamics*, rather than memorising the training signals, comes from driving it with pure sinusoids — inputs it never saw during training, since every training profile was a broadband ISO 8608 road. Sweeping a 5 mm sine from 0.5 Hz to 10 Hz and measuring the steady-state output/input amplitude ratio recovers the frequency response below.

![Frequency response](figures/frequency_response.png)

Two features stand out, and both are exactly what a two-mass suspension should produce:

- **The body isolates high-frequency road input.** The body/road ratio rises to a resonant peak near 2 Hz — the sprung-mass natural frequency — where road input is *amplified* roughly two-fold, then rolls off steeply, falling below 0.15 above 6 Hz. That roll-off is the entire purpose of a suspension: pass almost nothing at high frequency so the cabin stays still while the wheel absorbs the road.
- **The tire carries a second resonance.** The tire/road ratio shows the same low-frequency peak near 2 Hz, dips through an anti-resonance around 3.5 Hz, then rises to a second broad peak near 7.5–8 Hz. That upper peak is the wheel-hop mode — the unsprung mass bouncing on the tire's own stiffness — and the tire stays close to unity gain across a wide band because it tracks the road far more directly than the body does.

A physical quarter-car has precisely two resonances, one for each mass, and both appear here at sensible frequencies with the correct relative behaviour. The model was never given a spring rate, a damping coefficient, or a single sinusoid; it reconstructed the system's modal structure from road-elevation data alone.

**Reproducing the headline numbers:** the results table uses `SEQ_LEN = 1000`, `STRIDE = 5`, and `N_SAMPLES = None`, evaluated with `gru_bs128_s5_seq1000_100k.pt` (kept in `models/`).

---
## System & Method
 
- **Dataset:** Quanser Active Suspension benchmark, committed to the repository as `data/suspension_dataset.mat` and originally published on Zenodo (<https://zenodo.org/records/17232645>). Road profiles follow the ISO 8608 PSD formulation as multisine signals across five roughness classes A–E, 100,000 samples per class at 1 kHz — a 2 km road traversed at 20 m/s. Measured displacements span four periods; period 1 is discarded as transient and period 2 is used for training. Only the unsuppressed (full-signal) data is used.
- **Normalisation:** Global z-score, computed once across all classes and saved to `models/zscore_stats.pkl` so inference and evaluation denormalise with exactly the statistics the model was trained under. MinMax scaling was rejected because Class E amplitudes are roughly an order of magnitude larger than Class A, which squashed the smooth roads to nearly nothing.
- **Windowing:** `numpy.lib.stride_tricks.sliding_window_view` builds every 1,000-sample window in one vectorised pass, then a stride of 5 subsamples them. The target is the displacement at the *last* sample of each window, making this a causal many-to-one predictor — it never sees road ahead of the wheel. Windows are shuffled before a 90/10 train/validation split.
- **Architecture:** Two stacked GRU layers (hidden size 128) with ReLU between them, the final timestep's hidden state passed to three fully connected layers of width 200, then a linear output layer producing body and tire displacement (~256k trainable parameters).
- **Training:** MSE loss, NAdam optimiser, `ReduceLROnPlateau` (factor 0.9, patience 10, min LR 1e-16), batch size 128, up to 300 epochs with early stopping after 50 epochs without improvement. The best-validation checkpoint is saved to `models/`.
- **Evaluation:** Per-class RMSE, MAE, NRMSE (by both range and standard deviation) and correlation, plus out-of-distribution tests — speed bumps at several heights, a pothole, the sine sweep shown above, and a quarter-car model fitted to the network's own output as a physical sanity check.
---
 
## Tech Stack
 
- **Language:** Python (Jupyter Notebooks)
- **Deep learning:** PyTorch (GRU, NAdam, ReduceLROnPlateau)
- **Numerics and signals:** NumPy, SciPy (`loadmat`, `solve_ivp`, `lsim`, `minimize`)
- **Utilities:** scikit-learn, joblib (scaler persistence)
- **Visualization:** Matplotlib
---
 
## Repository Structure
 
```
.
├── notebooks/
│   ├── train_suspension_gru.ipynb       # Data loading, training, quarter-car comparison
│   └── evaluate_suspension_gru.ipynb    # Per-class metrics, frequency and bump tests
├── models/
│   ├── gru_bs128_s5_seq1000_100k.pt      # Checkpoint used for all reported results
│   └── zscore_stats.pkl                  # Normalisation statistics (written by training)
├── data/
│   └── suspension_dataset.mat            # Quanser benchmark data (tracked)
├── figures/                              # Plots used in this README
├── reports/                              # Written report and poster
├── requirements.txt
└── README.md
```
 
---
 
## How to Run
 
```bash
git clone https://github.com/PremYaragarla/Suspension-NN.git
cd Suspension-NN
pip install -r requirements.txt
jupyter lab         # Or Jupyter Notebook
```
 
The dataset ships with the repository at `data/suspension_dataset.mat`, so there is nothing to download — the notebooks run straight after a clone. Run the evaluation notebook to reproduce every result above from the committed checkpoint; run the training notebook to retrain from scratch.
 
**Requirements:** [numpy, scipy, torch, scikit-learn, joblib, matplotlib, jupyter, or `pip install -r requirements.txt`]
 
**Training hardware:** The model was trained on GTX 1660 Ti. Training can be done on a CPU, but it is VERY slow at full size. For a quick pass, set `N_SAMPLES = 10_000` in the configuration cell; leave it at `None` for the full 100,000 samples per class on a GPU.
 
---
 
## What I'd Improve Next
 
- **Fix the small-amplitude bias.** Class E dominates the MSE and pulls accuracy away from smooth roads (13.3 % vs 2.0 % NRMSE/std on the body). Weighting the loss per class, normalising each class independently, or training on relative rather than absolute error should all narrow the gap on Class A.
- **Add a physics-informed loss term.** An energy-balance or quarter-car residual penalty would push predictions toward physically consistent trajectories. The fitted quarter-car parameters already in the training notebook give a starting point, though the real system's nonlinearities may conflict with a simplified constraint, so this needs testing rather than assuming.
- **Benchmark against Hammerstein-Wiener.** The dataset ships with reference predictions from a classical nonlinear block model. Scoring the GRU against it on identical windows would turn "the errors are small" into "the errors are smaller than the standard baseline."
- **Move beyond synthetic roads.** Every profile here is ISO 8608 multisine. Measured road data would be the real test of whether the model learned suspension dynamics or the statistics of the generator.
- **Run it in the loop.** The end goal is an ECU that predicts displacement a few milliseconds ahead and adjusts damping accordingly — which means a fixed sample rate, a bounded inference budget, and testing against the Quanser rig in real time.