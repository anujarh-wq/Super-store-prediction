# Super Store Sales Prediction (KNN Regression)

A machine learning project that predicts **item-level outlet sales** for a retail chain using a K-Nearest Neighbors regressor, with an interactive Streamlit dashboard ("Retail Pulse") for single predictions and bulk CSV scoring.

## Project Structure

```
.
├── KNN_reg_outlet_sales.csv       # Dataset (8,523 rows)
├── KNN Regression.ipynb           # Preprocessing, training, tuning, evaluation
├── app.py                         # Streamlit dashboard
├── knn_regression_model.pkl       # Trained KNN model (k=7, Manhattan distance)
├── minmax_scaler.pkl              # Fitted MinMaxScaler
├── features.pkl                   # Ordered list of 35 model input features
├── requirements.txt               # Python dependencies
└── README.md
```

## Dataset

`KNN_reg_outlet_sales.csv` contains 8,523 product-outlet records across 1,559 unique products and 10 outlets.

| Column | Description |
|---|---|
| `Item_Identifier` | Unique product ID |
| `Item_Weight` | Product weight |
| `Item_Fat_Content` | Low Fat / Regular |
| `Item_Visibility` | Share of display area allocated to the product in the store |
| `Item_Type` | Product category (16 types) |
| `Item_MRP` | Maximum retail price |
| `Outlet_Identifier` | Unique store ID |
| `Outlet_Establishment_Year` | Year the outlet was established |
| `Outlet_Size` | Small / Medium / High |
| `Outlet_Location_Type` | Tier 1 / 2 / 3 |
| `Outlet_Type` | Grocery Store or Supermarket Type 1 / 2 / 3 |
| `Item_Outlet_Sales` | **Target**: sales of the product at the outlet |

## Methodology

1. **Missing values**
   - `Item_Weight` (1,463 missing): filled with the median (12.6)
   - `Outlet_Size` (2,410 missing): filled with the mode (Medium)
2. **Cleaning**: standardized inconsistent `Item_Fat_Content` labels (`LF`, `low fat` → Low Fat; `reg` → Regular).
3. **Encoding**
   - Ordinal: `Item_Fat_Content` (0/1), `Outlet_Size` (0–2), `Outlet_Location_Type` (0–2)
   - One-hot: `Item_Type` (drop-first), `Outlet_Identifier`, `Outlet_Type` (drop-first)
   - Dropped: `Item_Identifier`, and the original categorical columns after encoding
4. **Features**: 35 input features in total (see `features.pkl`).
5. **Scaling**: `MinMaxScaler`.
6. **Model**: `KNeighborsRegressor`, tuned with `GridSearchCV` and `RandomizedSearchCV` over `n_neighbors` (2–24) and `p` (1, 2).
7. **Baselines compared**: Decision Tree and Random Forest regressors.

## Results

Metrics below are taken from the notebook (R² on the test set):

| Model | R² |
|---|---|
| KNN (default, k=5) | 0.498 |
| KNN (k=15, p=1) | 0.579 |
| KNN (k=7, p=1), 80/20 split | 0.550 |
| Decision Tree | 0.202 |
| Random Forest | 0.558 |

Baseline KNN error on the test set: RMSE ≈ 1188, MAE ≈ 838.

The final saved model is `KNeighborsRegressor(n_neighbors=7, p=1)`.

## Streamlit Dashboard

`app.py` has three modes:

- **Single Sales Predictor**: enter pricing, product, and outlet values and get a predicted sales figure with a gauge chart.
- **Batch Prediction (CSV)**: upload a CSV containing all 35 feature columns, get predictions, a distribution histogram, summary metrics, and a downloadable results file.
- **Model Analytics**: a short explanation of the KNN approach.

## Getting Started

### Prerequisites

- Python 3.9+

### Installation

```bash
pip install streamlit pandas numpy scikit-learn plotly matplotlib seaborn jupyter
```

> The pickles were created with scikit-learn 1.7.2. Use the same version to avoid compatibility warnings: `pip install scikit-learn==1.7.2`

### Run the dashboard

```bash
streamlit run app.py
```

Keep the three `.pkl` files in the same folder as `app.py`.

### Retrain the model

Run all cells in `KNN Regression.ipynb`.

### Batch CSV format

The uploaded CSV must contain the 35 columns listed in `features.pkl`, already encoded (not the raw categorical columns):

`Item_Weight, Item_Fat_Content, Item_Visibility, Item_MRP, Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type`, the `Item_Type_*` dummies, the `Outlet_Identifier_*` dummies, and the `Outlet_Type_*` dummies.

## Known Issues & Future Improvements

- **`requirements.txt` is incorrect.** The file in the project (`requirements.txt.txt`) contains an unrelated HTML snippet instead of dependencies. Rename it to `requirements.txt` and replace its contents with the package list above.
- **Scaler mismatch (needs verification).** The app applies `MinMaxScaler` before prediction, but the final model cell in the notebook (`knn_reg2`) was trained on unscaled `X`. Confirm which data the saved model was trained on, since KNN is distance-based and a mismatch will distort predictions.
- **"Feature Weightage Profile" chart is random.** The Model Analytics tab plots randomly generated values, not real feature importances. Replace with permutation importance or remove it.
- **Outlet inputs in the single predictor** are entered as raw numeric fields for the one-hot columns; a dropdown for item type, outlet ID, and outlet type would be more user-friendly.
- **Currency**: the app shows ₹ while the dataset has no stated currency.
- R² of ~0.55–0.58 leaves room for improvement: try gradient boosting (XGBoost/LightGBM), log-transform the target, and engineer features such as outlet age and price per unit weight.

## Tech Stack

Python · pandas · NumPy · scikit-learn · Plotly · Matplotlib · Seaborn · Streamlit
