# TabDiffuse

Score-based diffusion for privacy-aware synthetic tabular data.

TabDiffuse learns the distribution of a real table and draws new rows with a variance-preserving score-based diffusion model. The same run measures statistical fidelity, empirical privacy risk, and downstream machine-learning utility. Synthetic data is **not** treated as automatically private. Privacy results are diagnostics with editable thresholds, not a differential-privacy guarantee.

## Problem statement

Sharing or training on raw tabular records exposes the people or organizations those rows describe. Classical copies and simple noise do not provide a measured trade-off between utility and disclosure risk. This project asks whether a score-based diffusion model can produce mixed-type synthetic rows that preserve useful structure, and how that sample compares with simpler generators on quality, utility, and empirical privacy risk.

## Objectives

- Ingest CSV, Excel, and Parquet tables locally.
- Profile types, missingness, duplicates, cardinality, and likely targets, without dropping columns silently.
- Train a score network for a variance-preserving SDE on numerical features and an explicit categorical representation.
- Generate synthetic rows, invert preprocessing, and optionally apply declared constraints.
- Report distribution, correlation, privacy, and train-on-synthetic / test-on-real utility metrics computed from the data.
- Save checkpoints, preprocessing metadata, HTML/PDF reports, and experiment history so a run can be repeated.

## Research questions

**RQ1.** Can score-based diffusion models generate synthetic tabular data that preserves the statistical characteristics of real data?

**RQ2.** How does synthetic data compare with real data in terms of feature distributions and correlations?

**RQ3.** Can synthetic data provide useful training data for downstream machine-learning tasks?

**RQ4.** What empirical privacy risks remain in generated synthetic records?

**RQ5.** How does TabDiffuse compare with simpler synthetic-data generation baselines?

## Architecture

```
User
  -> Streamlit UI
  -> ingestion, validation, profiling
  -> train / validation / holdout split
  -> preprocessing (fit on the training split only)
  -> variance-preserving score model
  -> reverse SDE or probability-flow sampling
  -> inverse transform and optional constraints
  -> quality, privacy, and ML utility
  -> experiment record, HTML/PDF report, downloads
```

The diffusion model estimates the score ∇ₓ log pₜ(x). Training uses weighted denoising score matching,

‖ σₜ sθ(xₜ, t) + z ‖²,

with a linear or cosine β schedule. Sampling integrates the reverse SDE or the probability-flow ODE. Categorical columns are one-hot encoded, or placed on a fixed spherical embedding and decoded by nearest prototype when cardinality is high. Integer category codes are not treated as numeric magnitudes.

The downstream holdout is never used to fit the preprocessor or the diffusion model.

## Technology stack

Python 3.12, PyTorch, NumPy, Pandas, scikit-learn, SciPy, Matplotlib, Plotly, Streamlit, PyYAML, Pydantic, Joblib, PyArrow, and ReportLab. Optional packages (CTGAN, MLflow, UMAP, SDMetrics) are listed in `requirements-optional.txt` and are not required for the core pipeline. The model runs locally. No paid API is used, and uploaded tables are not sent to an external service.

## Installation

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt
```

On macOS or Linux, create the virtual environment with `python3.12 -m venv .venv` and activate it with `source .venv/bin/activate`.

Optional extras:

```powershell
python -m pip install -r requirements-optional.txt
```

Copy `.env.example` to `.env` if you want to override the log level, seed, or MLflow flag. Existing environment variables are not replaced.

## Dataset preparation

The bundled file `data/raw/sample_customer_data.csv` is **simulated demonstration data**, not a customer extract. Regenerate it with:

```powershell
python -m src.data.sample_data
```

You can also upload your own CSV, Excel, or Parquet file in the app. The uploader shows the file name, size, shape, memory use, dtypes, missing values, duplicate rows, and unique counts. Profiling then warns about high missingness, high cardinality, constant columns, ID-like columns, class imbalance, and duplicate rows. Those warnings do not delete data.

For a first run on CPU, use **Demo Mode** or a few hundred to a few thousand rows. The code is aimed at tables up to about 100,000 rows and 50–100 columns, with evaluation subsamples disclosed when they are used.

## Running the application

```powershell
streamlit run app.py
```

A step-by-step guide for someone using the screens is in `docs/USER_GUIDE.md`.

Sidebar pages: Home, Dataset Upload, Data Profiling, Preprocessing, Model Configuration, Training, Synthetic Generation, Quality Evaluation, Privacy Evaluation, ML Utility, Experiment Comparison, Reports, Settings.

Demo flow from Home: **Load demonstration dataset**. The banner reads “Demo Mode – Reduced training configuration”. Then profile, preprocess, train, generate, evaluate, and write a report. A headless demo is also available:

```powershell
python -m src.demo_pipeline
```

## Training

Configure the SDE schedule, step grid, hidden size, depth, dropout, epochs, batch size, learning rate, weight decay, seed, and CPU or CUDA. **Use recommended configuration** picks a smaller network for small tables. Training runs in a background thread and records real loss each epoch, so the chart can be paused at an epoch boundary or stopped between batches. Each run receives a new id such as `EXP-2026-0001` and does not overwrite an older experiment. Checkpoints store weights, optimizer state, epoch, configuration, preprocessing metadata, seed, schema, and loss history. **Resume from checkpoint** continues that same id after you raise the epoch count.

If CUDA is missing or GPU memory is insufficient, the message asks you to switch to CPU or to reduce the batch size and hidden dimension.

## Synthetic data generation

Choose 1,000, 5,000, 10,000, 50,000, or a custom count, or an augmentation ratio of 10%, 25%, 50%, 100%, or 200% relative to the uploaded table. Sampling steps, temperature, and seed are configurable. Optional class balancing and column conditioning re-inject the encoded condition during the reverse process and set it exactly at the end. Constraints in `data/constraints.example.yaml` are applied only when you enable them. Reject is the default action. Clipping is counted and never silent.

Downloads: CSV and Parquet under `data/synthetic/<experiment id>/`.

## Evaluation

Quality compares the training split with the synthetic sample:

- numerical: mean, median, standard deviation, quantiles, Kolmogorov–Smirnov, Wasserstein, histogram Jensen–Shannon, and PSI
- categorical: frequency gap, total variation, Jensen–Shannon, missing-rate difference
- Pearson and Spearman correlation matrices and their difference
- composite similarities with the formulas shown in the UI
- PCA, t-SNE, and UMAP (if installed) as qualitative plots only

## Privacy

The privacy page reports exact row matches (with a documented rounding rule), nearest-neighbor distance versus real-to-real distance, training-versus-holdout membership proximity, and an attribute-disclosure probe from quasi-identifiers. Labels are “Low observed risk”, “Moderate observed risk”, and “Higher observed risk”. Cutoffs live in Settings and are stored with the report.

These metrics are empirical privacy-risk diagnostics and do not constitute a formal differential privacy guarantee.

## ML utility

Three setups are scored on the real holdout:

- train on real
- train on synthetic
- train on real plus synthetic

Classification reports accuracy, precision, recall, F1, and ROC-AUC. Regression reports MAE, RMSE, and R². The app does not declare a winner.

## Baseline comparison

On the Experiment Comparison page you can run:

1. random row resampling
2. a Gaussian on numeric columns with independent categorical marginals
3. a Gaussian copula
4. CTGAN, if the optional package is installed and the checkbox is enabled

The table shows quality scores, an exact-match rate, a distance ratio, a single utility probe, and generation time.

## Project structure

```
app.py
config.yaml
src/
  config/          settings
  data/            load, validate, profile, split, sample data
  preprocessing/   numeric, categorical, dates, traceable inverse
  diffusion/       schedule, score network, loss, sampler, trainer
  generation/      sampler wrapper, constraints, postprocess
  evaluation/      quality, privacy, utility, baselines
  visualization/   Plotly figures
  reporting/       HTML and PDF
  services/        experiments, training job, pipeline
  ui/              Streamlit pages
tests/
notebooks/
data/raw/sample_customer_data.csv
```

## Configuration

`config.yaml` controls the seed, split ratios, preprocessing, diffusion schedule, network size, training, generation, evaluation sample caps, privacy thresholds, the demo configuration, baselines, and MLflow. The Model Configuration page edits the run that will be trained. Settings edits the privacy label cutoffs for the current session.

## Testing

```powershell
python -m pytest
```

Tests cover loading, profiling, preprocessing round trips, the noise schedule, the score network, sampling, a short training loss decrease, constraints, privacy matches, utility, report files, and a small end-to-end train-and-generate path.

## Limitations

- The categorical representation is continuous (one-hot or fixed embeddings), not multinomial diffusion.
- Exact-match privacy uses a documented rounding rule, because inverse scaling is floating point.
- Privacy labels are heuristics. They are not anonymity, k-anonymity, or differential privacy.
- Projection plots are qualitative.
- Utility models use fixed, small hyperparameters and may subsample very large tables. The cap is recorded.
- Training is local and single-user. A pause takes effect at the next epoch boundary.
- CTGAN, UMAP, and MLflow are optional.
- Date parts that cannot form a calendar date are moved onto a real day in that month and counted. Numeric samples are kept inside the training column range and counted.

## Future enhancements

- Multinomial diffusion for categorical coordinates.
- Formal differential privacy accounting.
- Larger conditional architectures and calibrated rare-category models.
- Deeper baseline tuning, including a required CTGAN environment.
- Distributed training for tables beyond the current single-machine target.

## Experiment outputs

A finished run can produce a synthetic CSV, a synthetic Parquet file, checkpoints, preprocessing JSON, training history, quality metrics, privacy diagnostics, utility metrics, figures, an HTML report, a PDF report, and `artifacts/experiments/<id>/experiment.json` with the seed, Python version, package versions, and configuration needed to repeat the setup.
