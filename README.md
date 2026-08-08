<div align="center">

# EV Battery Health Prediction — State of Health with Linear Regression

**A beginner-friendly machine learning project that predicts the State of Health (SoH) of an EV battery from everyday sensor data — using Linear Regression.**

A battery starts at ~100% health and is typically considered *end of life* at **80%**. This project teaches how to estimate the remaining health from cycle number, voltage, current, temperature and more — with an accuracy of about **±1 percentage point**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-LinearRegression-F7931E?style=for-the-badge&logo=scikitlearn)](https://scikit-learn.org/)
[![R² score](https://img.shields.io/badge/R%C2%B2-0.9967-2e9b4a?style=for-the-badge)](https://scikit-learn.org/)
[![RMSE](https://img.shields.io/badge/RMSE-1.08%20pp-D95D3F?style=for-the-badge)](https://scikit-learn.org/)
[![Live site](https://img.shields.io/badge/Try%20it%20live-EV%20Battery%20Lab-3FBF5F?style=for-the-badge)](https://abhiramgangadharan07-crypto.github.io/ev-battery-health-site/)

</div>

> **Try the interactive version:** this model powers the [EV Battery Lab](https://abhiramgangadharan07-crypto.github.io/ev-battery-health-site/) website — a fully static site that runs the prediction in your browser, with a 3D battery visualisation and all 2,000 samples as a rotatable point cloud.

---

## Table of contents

- [The research problem](#research-problem)
- [The dataset](#dataset)
- [Methodology](#methodology)
- [Why Linear Regression fits this problem](#why-linear-regression-fits-this-problem)
- [Results](#results)
- [What you'll learn](#what-youll-learn)
- [How to run](#how-to-run)
- [Project structure](#project-structure)
- [Author](#author)

## Research problem

An EV battery degrades every time it is charged, discharged, and thermally stressed. The **State of Health (SoH)** measures how much of its original capacity a battery still has:

```
SoH (%) = (current maximum capacity / nominal capacity when new) × 100
```

A battery starts at ~100% and is typically considered "end of life" around **80%**. Being able to estimate SoH from everyday sensor data (temperature, voltage, current, number of charge cycles) is important for **three practical reasons**:

1. **Safety** — a degraded battery is more likely to overheat, swell, or fail unpredictably. Monitoring SoH helps detect dangerous cells early.
2. **Resale value** — the biggest single factor determining a used EV's price is the remaining health of its battery. Buyers and sellers both need an honest estimate.
3. **Maintenance planning** — knowing how fast a battery is losing capacity lets owners and fleet operators schedule replacements or warranty claims before the battery leaves the vehicle.

SoH estimation is a textbook **regression** problem: the target is a number, and we want to predict it from a set of measurable features.

## Dataset

- **Name:** Battery State of Health Dataset
- **Source:** Kaggle — https://www.kaggle.com/datasets/freshersstaff/battery-state-of-health-dataset
- **Contents:** 2,000 multi-cycle battery samples. Features: cycle number, voltage, current, temperature, charge/discharge time, internal resistance, capacity, ambient humidity, C-rate, and the health target.
- **Target:** `SOH` (State of Health, %) — 100% when new, down to ~28% when heavily aged.

> **Dataset note:** the originally assigned dataset (*Electric Vehicle (EV) Battery Degradation & Charge*, ~10,000 NMC/LFP samples) was taken offline by its owner in 2026 — every Kaggle route (web page, API download, consent flow) returns 404/403. To keep the project fully runnable, this project uses the **"Battery State of Health Dataset"** above — a directly comparable SoH regression dataset. The methodology, notebook and code are otherwise unchanged.
>
> Download the CSV from the link above (Kaggle account required) and place it inside `data/` as `EV_Battery_Data.csv`. The notebook auto-detects any CSV in that folder.

## Methodology

The pipeline in `analysis.ipynb` follows these steps:

| Step | What we do | Why |
| ---- | ---------- | --- |
| **Load** | `pd.read_csv()` + `df.head()` | Verify the data was read correctly |
| **Inspect** | print `df.dtypes` and column names | Check data types before modelling |
| **Clean** | auto-detect the SoH/SOH target and drop rows with missing values | Models cannot learn from empty cells; this CSV is complete (2,000 of 2,000 rows kept) |
| **Encode** | drop ID columns (`BatteryID`, `BatchID`) and one-hot encode any text columns with `pd.get_dummies(...)` | sklearn only accepts numbers; this dataset's features are all already numeric |
| **Split** | `train_test_split(X, y, test_size=0.2, random_state=42)` | Train on 80%, test on the 20% the model never saw |
| **Train** | `LinearRegression().fit(X_train, y_train)` | Find the coefficients that best explain SoH |
| **Evaluate** | `r2_score` and RMSE on the test set | Measure how well the model generalises |
| **Visualise** | scatter plot: actual vs predicted SoH + diagonal reference line | Sanity-check the model visually |

### Why Linear Regression fits this problem

- Battery capacity fades **approximately linearly** with cumulative charge cycles over the early/mid life of the pack, so a linear model is a theoretically sound baseline for SoH.
- It is **interpretable**: the fitted coefficients tell us the direction and size of each factor's effect — e.g. the coefficient on `Cycle` is −0.032, so each additional charge/discharge cycle reduces the expected SoH by about 0.03 percentage points.
- It is fast, has almost no hyper-parameters, and is a perfect benchmark against more complex models.

## Results

The model was trained on **1,600** of the 2,000 samples and evaluated on the **400** unseen ones. The values below were generated by running `analysis.ipynb`.

| Metric | Value | What it means |
| ------ | ----- | ------------- |
| **R² (R-squared)** | **0.9967** | The model explains 99.67% of the variation in battery health — an excellent fit. |
| **RMSE** | **1.0804** | The average prediction error is about **1.08 percentage points** of SoH. |

**In plain language:** given the cycle- and electrical readings of a battery, the model can estimate its State of Health to within ~±1 percentage point, and it captures nearly all of the differences between batteries.

**Why it performs so well:** in this dataset, State of Health declines almost *linearly* with charge/discharge cycles (each cycle costs about 0.03 percentage points), and the linear model captures that dominant trend while the remaining features (internal resistance, temperature, current, C-rate, capacity, ...) refine the estimate. This is exactly the regime where Linear Regression is the right tool — simple, fast, and interpretable.

### Evaluation plot

![Linear Regression: actual vs predicted SoH](images/result_plot.png)

The red dashed diagonal is the **perfect prediction** line (`y = x`). Each dot is one battery in the test set; the closer the dots are to the diagonal, the more accurate the model.

## What you'll learn

By working through `analysis.ipynb` you'll practice the standard end-to-end ML workflow:

1. Reading and inspecting a real-world CSV with pandas
2. Cleaning data (missing values) and encoding features for sklearn
3. Splitting data into honest train/test sets (`random_state=42` for reproducibility)
4. Fitting a `LinearRegression` model in one line
5. Evaluating with **R²** and **RMSE** — and *interpreting* what the numbers mean
6. Building a diagnostic scatter plot with matplotlib

## How to run

1. **Install Python 3.9+** (or use a free Jupyter environment).
2. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset** from Kaggle (link above) and place the CSV inside the `data/` folder.
4. **Open the notebook**:

   ```bash
   jupyter notebook
   ```

   then open `analysis.ipynb` and select **Run All** from the `Cell` menu.

5. Optional — run headlessly and convert it to an HTML report:

   ```bash
   jupyter nbconvert --to notebook --execute --inplace analysis.ipynb
   ```

   The plot is saved automatically to `images/result_plot.png`.

## Project structure

```
ev-battery-health-prediction/
├── README.md              <- this file
├── analysis.ipynb        <- the full pipeline (explained cell by cell)
├── requirements.txt       <- Python dependencies
├── .gitignore             <- keeps data/ out of Git
├── data/                  <- (your download) the CSV goes here
└── images/
    └── result_plot.png    <- generated by the notebook
```

## Author

**Abhiram Gangadharan** — B.Tech Electronics & Communication Engineering, Sree Buddha College of Engineering.

[![GitHub](https://img.shields.io/badge/GitHub-abhiramgangadharan07--crypto-1C3A2E?style=for-the-badge&logo=github)](https://github.com/abhiramgangadharan07-crypto)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-abhiram--gangadharan-0A66C2?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/abhiram-gangadharan-a6282a379)
[![Kaggle](https://img.shields.io/badge/Kaggle-abhiramgangadharan07-20BEFF?style=for-the-badge&logo=kaggle)](https://www.kaggle.com/abhiramgangadharan07)
[![Email](https://img.shields.io/badge/Email-abhiramgangadharan07@gmail.com-D95D3F?style=for-the-badge&logo=gmail)](mailto:abhiramgangadharan07@gmail.com)
