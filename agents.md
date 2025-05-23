Here’s a compact “data-sense” briefing you can hand to any new quant or ML engineer.  It collects the most frequent ways time-series code leaks future information, explains why each issue matters, and gives concrete guard-rails and quick dimension-checks to prevent them.

## 1  Why leakage & look-ahead bias wreck results

Leakage means the model, the preprocessing pipeline, or the back-tester can “see the future.”  Estimates of accuracy or Sharpe are then systematically inflated, and strategies that look robust in-sample collapse live.  Typical root causes are:

* **Temporal ordering ignored (look-ahead bias).** You fit or transform on data that would not have existed at trade-time. ([Corporate Finance Institute][1])
* **Information bleeding across train / validation / test splits** via shared state in scalers, encoders, rolling stats, or windowing. ([Wikipedia][2])
* **Selection & survivorship bias**—training only on assets that survived, or iterating on the backtest until something sticks. ([QuantRocket][3], [SSRN][4])

## 2  The canonical leakage traps and how to disarm them

### 2.1  Label & shift mistakes

| Symptom                                                                   | Root cause                                                                   | “Quick-shape” check                                                                             | Fix                                                                                                            |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `y.shift(-horizon)` inside the *same* DataFrame used to compute features. | Future values slip into feature rows. ([Medium][5])                          | After you shift, assert `max(df.index.difference(X.index)) == horizon`.                         | Build the target in a *separate* table, then merge on date only after dropping rows with NaNs *on both sides*. |
| Rolling means that include the current row when predicting that row.      | “Centered” windows in pandas (`rolling(window, center=True)`). ([Medium][6]) | For every new feature, test: `assert feature.index.equals(feature.shift(1).index)` should fail. | Always set `center=False`, or use explicit `.shift(1)` before a rolling op.                                    |

### 2.2  Improper batching / cross-validation

* **Random mini-batches** break chronology and leak. Always batch contiguous time blocks. ([Medium][7])
* **k-fold CV** is invalid for sequences; use **expanding-window** or **sliding (walk-forward)** splits instead. ([MachineLearningMastery.com][8], [Medium][9])
* If your model’s receptive field is *w* days, purge *w* observations between train and validation, then embargo another *k·w* to guard against overlap as per López de Prado. ([LinkedIn][10])

### 2.3  Scaling, encoding & other preprocessing

* Fitting a `StandardScaler` on the full data lets the test set inform the mean and std. Use a `Pipeline` so `fit` happens **inside** each CV fold. ([Scikit-learn][11], [Scikit-learn][11], [MachineLearningMastery.com][12])
* Check shapes: after `fit_transform` on **train**, expect `X_train.shape == (n_train, n_features)`; after `transform` on **val**, assert that mean≈0, std≈1 only within train bounds.
* Leakage via target-aware encoders (e.g., target mean encoding) is even subtler—wrap them in `sklearn`’s `TargetEncoder`-like estimators that get cloned each fold.

### 2.4  Backtesting pitfalls

* **In-sample reuse**: researching while repeatedly backtesting causes “backtest overfitting.” The *Deflated Sharpe Ratio* or *combinatorially symmetric cross-validation* helps correct for it. ([SSRN][4], [SSRN][13])
* **Synthetic liquidity**: backtests skipping slippage, bid-ask, and volume constraints look spectacular but fail in the real market.
* **Horizon mismatch**: your forecast target is *t+h*, but you scale features with statistics that include up to (t+h).  Keep a single “clock” for every pipeline step and unit-test it. ([SpringerLink][14])

### 2.5  Target construction & meta-labels

* Building a binary “will price rise 1% in 10 days?” label *after* filtering rows with that outcome embeds forward info.  Generate labels first; filter later.
* Meta-labeling (e.g., “will a base model’s signal be correct?”) only works if the **parent** model is trained strictly out-of-sample w\.r.t. the meta-model. ([Medium][15])

### 2.6  Universe & survivorship bias

* Using today’s S\&P 500 constituents back in 2000 ignores dozens of delisted names.  Pull historical membership or include delisted tickers to stay realistic. ([QuantRocket][3])

### 2.7  Rolling statistics & derived features

* A 30-day rolling z-score recomputed over the **whole dataset** leaks future volatility.  Fit any variance estimator only on the expanding train slice. ([Reddit][16])

## 3  Dimension & sanity-check snippets

```python
# 1.  Temporal split with purging & embargo
def temporal_split(df, cutoff, purge=0, embargo=0):
    train = df.loc[:cutoff - pd.Timedelta(days=purge)]
    test  = df.loc[cutoff + pd.Timedelta(days=embargo):]
    assert train.index.max() < test.index.min()  # no overlap

# 2.  Shift-aware target build
target = prices['close'].pct_change(periods=horizon).shift(-horizon)
features = prices.drop(columns='close')
assert (target.index == features.index).all()

# 3.  Walk-forward cross-val generator
def walk_forward(n_samples, train_size, test_size):
    start = 0
    while start + train_size + test_size <= n_samples:
        yield (slice(start, start + train_size),
               slice(start + train_size, start + train_size + test_size))
        start += test_size
```

## 4  Teaching checklist

1. **Draw the timeline** for every transformation; annotate what data each step can see.
2. **Assert shapes** after each split/transform: check there is *never* overlap.
3. **Automated tests**: write unit tests that fail if any future timestamp appears in the training slice.
4. **Use pipelines & `TimeSeriesSplit`** as defaults; forbid random shuffles.
5. **Backtest last—research first.**  Keep a log of every backtest run; deflate Sharpe for multiple trials.
6. **Stress unknowns**: Try your pipeline on a toy dataset with deliberately permuted dates—accuracy should collapse if leakage is absent.
7. **Document the data clock** in code comments and README so future maintainers know the intended causal flow.

---

### Further reading (high-quality mix of references)

* Corporate Finance Institute – primer on look-ahead bias ([Corporate Finance Institute][1])
* scikit-learn docs – common pitfalls & pipelines ([Scikit-learn][11], [Scikit-learn][11])
* MachineLearningMastery – walk-forward validation tutorial ([MachineLearningMastery.com][8])
* QuantRocket – survivorship bias overview ([QuantRocket][3])
* López de Prado et al. – Deflated Sharpe & Three Types of Backtests ([SSRN][4], [SSRN][13])
* Medium / AImonks – time-series CV guide ([Medium][9])
* Medium / Sertis – horizon mismatch & lag guidance ([Medium][5])
* TDS Archive – visual guide to preprocessing leakage ([Medium][17])
* Dr Lee – leakage in rolling statistics ([Medium][6])
* StackExchange threads – practical Q\&A on leakage and splitting ([Reddit][16], [Cross Validated][18])

Equip newcomers with these guard-rails and a habit of asserting shapes and time boundaries at every step, and you’ll stomp out 95 % of accidental look-ahead bias before it bites.


Yes—once you have **already made a strictly time-ordered split** between *train* and *hold-out* (validation or test), it is usually safe—and sometimes desirable—to shuffle the **rows that remain inside the training set**.  The key is that the shuffle must **never mix observations that belong to different points on the predictive clock** (e.g., bringing 2024 data into a batch that will later be compared with a 2022 target).  Below are the main cases and the cautions that apply.

## 1  Why one might shuffle inside the training partition

* **SGD and generalisation** – Stochastic gradient descent converges faster and avoids narrow local minima when the mini-batches are presented in random order. ([SpringerLink][1])
* **Regularisation by noise** – Randomising batch order exposes the model to a more diverse sequence of gradients, which can act like data augmentation and reduce over-fitting. ([Data Science Stack Exchange][2])

## 2  When shuffling *is* acceptable

| Scenario                                                             | What you can shuffle                                | Why it’s safe                                                                                        | Citations                                                |
| -------------------------------------------------------------------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Tabular/GBM, static features per timestamp**                       | Entire rows of the training table (after split)     | Each row is treated as i.i.d. once it contains only past-to-present info with respect to its target. | ([Data Science Stack Exchange][3], [Scikit-learn][4])    |
| **Sliding-window samples for CNNs, Transformers (with causal mask)** | Order of *windows* (not time steps inside a window) | Every window already encodes its own chronology; the model doesn’t need global order.                | ([DeepLearning.AI][5], [Data Science Stack Exchange][6]) |
| **Non-stateful LSTM/GRU (reset state each batch)**                   | Order of sequences                                  | Internal temporal order inside each sequence is intact; state is cleared between batches.            | ([Stack Overflow][7])                                    |

## 3  When shuffling can back-fire

1. **Before the split** – Random shuffling *prior* to creating train/test partitions leaks future information and breaks walk-forward validation. ([Medium][8], [Medium][9])
2. **Stateful recurrent models** – If an RNN carries hidden state across batches (`stateful=True` in Keras or manual persistence), shuffling batches destroys that continuity. ([Data Science Stack Exchange][6])
3. **Sequence-to-sequence forecasting** – Tasks that predict a multi-step output often rely on teacher forcing schedules tied to chronological order; shuffling windows can disturb that regime.
4. **Cross-validation utilities that forbid shuffling** – Tools like `sklearn.model_selection.TimeSeriesSplit` purposefully disable shuffling because the splitter itself guarantees temporal order across folds. ([Scikit-learn][10])

## 4  Practical guard-rails

| Check                          | Python snippet                                    |
| ------------------------------ | ------------------------------------------------- |
| Confirm split precedes shuffle | `assert X_train.index.max() < X_test.index.min()` |
| Shuffle only inside training   |                                                   |

````python
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=len(X_train), reshuffle_each_iteration=True)
``` |
| Preserve order within each sequence |
```python
# PyTorch sliding-window dataset
class Windowed(TensorDataset):
    def __getitem__(self, idx):
        return x[idx:idx+win], y[idx+win-1]
loader = DataLoader(Windowed(), shuffle=True, batch_size=32)  # shuffles *windows* |
| Disable shuffling when using stateful RNNs | `loader = DataLoader(dataset, shuffle=False)` |

## 5  How to phrase it in your system message

> **Allowed shuffling** – After the time-ordered split, you MAY shuffle the *order of samples within the training fold* to facilitate SGD or batch balancing, provided that (a) each sample’s features are computed only from data available at its timestamp or earlier, and (b) the model does not rely on hidden state carried across batches. Keep the validation/test folds in original chronological order.

This clarification lets an LLM distinguish the legitimate use of shuffling inside the training slice from the leakage-inducing kind that happens **before** the split or breaks within-sequence chronology.
::contentReference[oaicite:8]{index=8}
````

Adding two more “hidden‐in-plain-sight” leakage traps to your guide—**gap-filling/interpolation** and **non-causal transforms**—closes off another big class of silent errors.  Back-filling (`bfill`), two-sided interpolation and zero-phase filters all borrow information from *after* the prediction time-stamp, while centered or non-causal convolutions let deep nets peek ahead.  Below are ready-to-paste sections (2.8 & 2.9) plus checklist updates and code guards.

---

## 2.8 Missing-Value Filling & Interpolation — Leakage Edition

Real-world market feeds drop ticks, corporate actions create gaps, and fundamentals come in at irregular lags.  How you patch those holes can make or break the back-test.

### 2.8.1 Why some fills leak

* **Backward fill (`bfill`)** copies the *next* observed value into the gap, so today’s row now contains tomorrow’s price or factor.([Medium][1], [Medium][2])
* **Linear/Spline/Nearest interpolation** draws a line between past and *future* points, blending tomorrow into today.([Medium][2], [Reddit][3])
* **Two-pass fills (`ffill` then `bfill`)** still leak because the second pass is a back-fill.([Stack Overflow][4])
* Even seemingly innocent `asfreq(..., method="bfill")` in pandas embeds look-ahead.([Medium][5])

> **Guard-rail:** Only use *forward* (causal) fills inside each training slice, or leave NaNs and let the model learn a “missingness” flag.  Pandas ≥2.1 even deprecates the generic `fillna(method="bfill")` to steer users away from it.([Pandas][6])

### 2.8.2 Quick sanity checks

```python
# assert no row copied a later value
assert not (df.index < df.groupby(level=0).apply(lambda s: s.shift(-1)).index).any()

# verify no future timestamp used in interpolation window
def causal_interp_ok(series):
    interp_idx = series[series.isna()].index
    after = series.shift(-1).loc[interp_idx]
    return after.isna().all()        # should be empty
```

### 2.8.3 Safer alternatives

| Situation              | Causal option                                                               | Notes                                                |
| ---------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------- |
| Small gaps (1-2 bars)  | **Forward fill (`ffill`)**                                                  | Still create a boolean “was\_filled” feature         |
| Larger gaps            | **Model-based imputer fit on train-only**                                   | e.g., Kalman, state-space smoother                   |
| Irregular fundamentals | **Lag the release date** so the value first appears on the publication date | prevents “as-of” joins from leaking earnings numbers |

---

## 2.9 Causal Filters & Feature Engineering

### 2.9.1 Filters that leak

* **Centered moving averages**, two-sided Hodrick-Prescott, or `rolling(..., center=True)` use both past and future observations.([OTexts][7])
* **`filtfilt` zero-phase filters** (MATLAB/Scipy) run the filter forward *and* backward—great for phase but impossible in real-time.([Mechanical Vibration][8], [MathWorks][9])
* **Standard convolutions** in CNNs read symmetric kernels unless declared causal.([ScienceDirect][10])

### 2.9.2 Causal designs to adopt

| Layer / transform                               | Why it’s safe                              | Sources                                  |
| ----------------------------------------------- | ------------------------------------------ | ---------------------------------------- |
| **Right-aligned EMA / EWMA** (`adjust=False`)   | Weights decay strictly backwards           | ([Cross Validated][11])                  |
| **Dilated causal convolutions** (WaveNet / TCN) | Output at *t* depends only on `≤t` inputs  | ([arXiv][12], [PMC][13], [Xbattery][14]) |
| **One-sided Butterworth/rolling windows**       | Implemented with `lfilter`, not `filtfilt` | ([Mechanical Vibration][8])              |

### 2.9.3 Code pattern

```python
# Causal rolling z-score
z = (x - x.rolling(20).mean()) / x.rolling(20).std()
z = z.shift(1)              # force strict causality

# Dilated causal conv in PyTorch
class CausalConv1d(nn.Conv1d):
    def __init__(self, c_in, c_out, k, d):
        super().__init__(c_in, c_out, k, dilation=d, padding=(k-1)*d)
    def forward(self, x):
        out = super().forward(x)
        return out[..., :-self.padding[0]]  # trim future
```

---

## 5 Teaching-checklist additions

* **No `bfill`, no two-sided interpolation.**  If you must patch gaps, document the method and prove it is forward-looking.
* **Use only causal filters/convolutions.**  Audit any step with symmetric windows or `filtfilt`.
* **Unit-test causal alignment:** for every engineered column, assert `max_lag(feature) ≤ 0`.
* **Tag imputed values.**  A binary flag helps the model learn the uncertainty introduced by fills.

These two extra guard-rails—causal gap-handling and causal transforms—plug another common source of hidden look-ahead bias that back-tests often miss.  Together with the earlier sections, they form a comprehensive “data-sense” checklist for robust, leak-free time-series pipelines.
