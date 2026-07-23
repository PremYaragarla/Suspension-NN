# Suspension Displacement Prediction — GRU
 
> A recurrent neural network that predicts vehicle body and tyre displacement directly from a road profile, learned from Quanser active-suspension bench data with no physical parameters supplied.
 
![Class E predictions vs ground truth](figures/classE_predictions.png)

---

## Overview

**The problem:** Predicting how a suspension responds to the road is a core vehicle-dynamics task, and the standard tool is the quarter-car model. It is cheap and intepretable, but it requires accurate mass/stiffness/damping values and it assumes all behavior is linear. Real suspensions have bushings, friction, and travel limits that introduce nonlinearity that a linear model cannot represent. The objective of this project is to predict body and tire displacement simultaneously from road elevation alone, without identifying any physical parameters.

**The approach:** A GRU-based recurrent network maps a sliding 1,000-sample window of ISO 8608 road elevation (1 second of road history at 1 kHz) to the body and tire displacment at the end of that window. Two stacked GRU layers compress the road history into a hidden state, and a three-layer MLP decodes it into the two displacements. I chose a GRU over a plain RNN because suspension response depends on road history over hundreds of milliseconds, which I would not be able to capture otherwise. Training uses all five roughness classes at once, with global z-score normalisation — MinMax scaling failed because Class E amplitudes are roughly an order of magnitude larger than Class A, so a shared MinMax range squashed the smooth roads to nearly nothing and resulted in poor accuracy on smoother roads. 

**Outcome:** Correlation with ground truth is ≥ 0.997 on every road class, and body RMSE stays between 0.020 mm (Class A) and 0.071 mm (Class E) against displacements measured in millimetres. The model also behaves sensibly on inputs it never saw in training — speed bumps, potholes, and sine sweeps all produce physically plausible responses, and fitting a quarter-car model to the network's own predictions recovers a consistent set of spring and damper constants.

---

## Results

Evaluated on the held-out period of each ISO 8608 class, denormalised back to metres before scoring.
 
| Class | Body RMSE | Tyre RMSE | Body *r* | Tyre *r* |
|---|---|---|---|---|
| A | 0.0204 mm | 0.0142 mm | 0.9970 | 0.9983 |
| B | 0.0234 mm | 0.0207 mm | 0.9993 | 0.9995 |
| C | 0.0293 mm | 0.0360 mm | 0.9996 | 0.9996 |
| D | 0.0482 mm | 0.0623 mm | 0.9998 | 0.9997 |
| E | 0.0709 mm | 0.1002 mm | 0.9999 | 0.9998 |

Absolute error grows with roughness, but *relative* accuracy moves the other way: correlation is weakest on Class A, the smoothest road. Class E's much larger displacements dominate the MSE during training, so the loss has little incentive to resolve Class A's small-amplitude detail. This shows up as a slight negative bias on the smoothest profiles and is the clearest target for the next round of work.

![Training and validation loss](figures/training_loss.png)

Training was stopped when the validation loss spiked rather than letting the early-stopping counter run out; the best checkpoint by validation MSE is the one kept.
 
![Predicted vs actual scatter](figures/classE_parity.png)
 
The parity plot shows the errors are unbiased and evenly spread rather than concentrated at the extremes of travel — the model is not simply clipping large excursions.

**Reproducing the headline numbers:** the results table uses `SEQ_LEN = 1000`, `STRIDE = 5`, and `N_SAMPLES = None`.

Dataset from this link:
https://zenodo.org/records/17232645

Scroll to the bottom, suspension_dataset.mat