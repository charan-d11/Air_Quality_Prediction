# 🌫️ Global Urban Air Quality — Machine Learning Project

A machine learning project that analyzes hourly air pollution data from 29 major cities worldwide to predict air quality index and detect hazardous pollution events.

---

## 📂 Dataset

- **Source:** [Kaggle — Global Urban Smog PM2.5 Hourly](https://www.kaggle.com/datasets/iconicwasil/global-urban-air-quality-and-pollution-time-series)
- **Size:** 254,736 rows × 13 columns
- **Coverage:** 29 cities globally, hourly readings

### Features Used

| Column | Description |
|--------|-------------|
| `PM10_ug_m3` | Particulate matter (≤10 µm) |
| `PM2_5_ug_m3` | Fine particulate matter (≤2.5 µm) |
| `Carbon_Monoxide_ug_m3` | CO concentration |
| `Nitrogen_Dioxide_ug_m3` | NO₂ concentration |
| `Ozone_ug_m3` | Ozone level |
| `Dust_ug_m3` | Dust concentration |
| `UV_Index` | Ultraviolet radiation index |
| `European_AQI` | Target for regression |
| `Hazardous_Event` | Target for classification (0/1) |

---

## 🎯 ML Tasks

### 1. 🔴 Hazardous Event Classification
Predict whether a given hour will have a hazardous pollution event.

- **Model:** Random Forest Classifier
- **Target:** `Hazardous_Event` (0 = Safe, 1 = Hazardous)
- **Result:** ~100% accuracy ⚠️ *(likely due to AQI-derived target — see notes)*

### 2. 📈 AQI Score Regression
Predict the European AQI numeric score from pollutant readings.

- **Model:** KNN Regressor (K=15)
- **Target:** `European_AQI`
- **R² Score:** ~0.83
- **RMSE:** Low — model explains 83% of AQI variance

---

## 📊 Visualizations

| # | Chart | Purpose |
|---|-------|---------|
| 1 | 🕸️ Radar Chart | Compare pollution profiles of top 6 cities |
| 2 | 🌡️ Heatmap | City × Pollutant intensity across all 29 cities |
| 3 | 🫧 Bubble Chart | RFC feature importance — bigger = more important |
| 4 | 📉 K-Elbow Curve | R² score vs K value with gradient fill |
| 5 | 🎯 Residual Plot | Actual vs Predicted AQI + error distribution |

---

## 🛠️ Tech Stack

- **Language:** Python 3.13
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Environment:** VS Code + Jupyter Notebook
- **Data Access:** kagglehub

---

## 🚀 How to Run

**1. Clone or download this project**
```bash
git clone  https://github.com/charan-d11/Air_Quality_Prediction.git
cd Air_Quality_Prediction
```

**2. Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn kagglehub
```

**3. Download the dataset**
```python
import kagglehub
path = kagglehub.dataset_download("iconicwasil/global-urban-air-quality-and-pollution-time-series")
```
Or manually download from Kaggle and place `global_urban_smog_pm25_hourly.csv` in the project folder.

**4. Open and run the notebook**
```bash
jupyter notebook Air_Quality.ipynb
```

---

## 📁 Project Structure

```
Air-Quality-Prediction/
│
├── data
|    |--global_urban_smog_pm25_hourly.csv  # Dataset (download from Kaggle)
|    
├── NoteBook
|    |--Air_Quality.ipynb        # Main notebook (EDA + Models + Visualizations)
└── README.md                # This file
```

---

## 📌 Key Findings

- **PM2.5 and PM10** are the most important features for predicting hazardous events
- **Lahore, Delhi, and Beijing** consistently show the highest pollution levels across all pollutants
- KNN with **K=15** gives the best balance of accuracy and generalization (R² = 0.8321)
- Increasing K beyond 15 gives diminishing returns (only +0.0003 improvement at K=20)

---

## ⚠️ Notes & Limitations

- RFC score of 100% suggests `Hazardous_Event` may be directly derived from `European_AQI` — a form of data leakage
- Dataset is synthetic/simulated — real-world performance may vary
- Time-series nature of data not fully exploited (random split used instead of time-based split)

---

## 👤 Author

**Durga Charan Mallick**
- 📍 Bhubaneswar, India
- 🛠️ Built with Python + scikit-learn

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).