# Crop Yield Predictor: Calculus & Momentum Optimization Engine

A production-grade, mathematically rigorous agricultural yield prediction platform built with **FastAPI**, **Pure NumPy Vector Calculus**, and **Jinja2**.

This platform demonstrates the role of **partial derivatives** and **multivariable calculus** in machine learning optimization, comparing **Standard Gradient Descent** with **Momentum Optimization** on your own CSV dataset.

---

## 1. Project Objective & Architecture

Predict agricultural crop yield ($\text{ton/ha}$) from environmental and soil features:
- **Rainfall** ($x_1$, $\text{mm}$)
- **Temperature** ($x_2$, $\text{°C}$)
- **Soil pH** ($x_3$)
- **Humidity** ($x_4$, $\%$)
- **Fertilizer usage** ($x_5$, $\text{kg/ha}$)
- *Dynamic custom features from any user CSV dataset*

```
crop_yield_momentum/
│
├── app.py                      # FastAPI backend, calculus engine, ML pipeline & API endpoints
├── dataset_config.json         # Configurable dataset path & feature/target column mappings
├── Procfile                    # Render process entry: uvicorn app:app --host 0.0.0.0 --port $PORT
├── render.yaml                 # Render Blueprint configuration for 1-click cloud deployment
├── .python-version             # Pins Python 3.11.8 for fast pre-compiled binary wheel installation
├── requirements.txt            # Python dependencies (fastapi, uvicorn, numpy, pandas, matplotlib)
├── .gitignore                  # Git ignore rules
├── README.md                   # Comprehensive documentation & mathematical derivations
├── run.py                      # Local desktop launcher (opens default browser automatically)
│
├── data/
│   └── crop_data.csv           # Active agricultural dataset
│
├── templates/
│   └── index.html              # Modern dashboard with KaTeX math suite & feature attribution
│
└── static/
    ├── css/
    │   └── style.css           # Custom dark research console styling
    └── js/
        └── main.js             # Dual input synchronization, live API client, math sandbox
```

---

## 2. Mathematical Foundations

### A. Linear Regression Hypothesis
$$\hat{y}^{(i)} = \mathbf{w}^T \mathbf{z}^{(i)} + b = \sum_{j=1}^n w_j z_{ij} + b$$
Where:
- $\hat{y}$ is predicted crop yield.
- $\mathbf{z} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$ is the standardized feature vector fitted strictly on the training set to prevent data leakage.
- $\mathbf{w} \in \mathbb{R}^n$ are feature weights learned via optimization.
- $b \in \mathbb{R}$ is the learned baseline bias.

---

### B. Mean Squared Error (MSE) Objective Function
$$L(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m \left(\hat{y}^{(i)} - y^{(i)}\right)^2 = \frac{1}{m} \sum_{i=1}^m \left(\sum_{j=1}^n w_j z_{ij} + b - y_i\right)^2$$

---

### C. Analytical Partial Derivatives & Gradient Derivation
Let the scalar residual for sample $i$ be:
$$e_i = (\hat{y}_i - y_i)$$

By the **Multivariable Chain Rule**:

#### 1. Differentiate outer quadratic loss with respect to residual $e_i$:
$$\frac{\partial L}{\partial e_i} = \frac{\partial}{\partial e_i} \left[ \frac{1}{m} e_i^2 \right] = \frac{2}{m} e_i = \frac{2}{m} (\hat{y}_i - y_i)$$

#### 2. Differentiate inner residual with respect to weight $w_j$:
$$\frac{\partial e_i}{\partial w_j} = \frac{\partial}{\partial w_j} \left( \sum_{k=1}^n w_k z_{ik} + b - y_i \right) = z_{ij}$$

#### 3. Chain Rule product for weight gradients:
$$\frac{\partial L}{\partial w_j} = \sum_{i=1}^m \frac{\partial L}{\partial e_i} \frac{\partial e_i}{\partial w_j} = \mathbf{\frac{2}{m} \sum_{i=1}^m (\hat{y}_i - y_i) z_{ij}}$$

In vectorized NumPy matrix notation:
$$\nabla_{\mathbf{w}} L = \mathbf{\frac{2}{m} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})}$$

#### 4. Partial derivative with respect to bias $b$:
$$\frac{\partial e_i}{\partial b} = 1 \implies \mathbf{\frac{\partial L}{\partial b} = \frac{2}{m} \sum_{i=1}^m (\hat{y}_i - y_i)}$$

---

### D. Exact Momentum Optimization Update
To accelerate descent along flat valleys and suppress lateral oscillations across steep walls, the optimizer maintains a parameter velocity vector $\mathbf{V}_t$:

$$\mathbf{V}_t = \mathbf{V}_{t-1} + (1 - \beta) \nabla L(\boldsymbol{\theta}_{t-1})$$
$$\boldsymbol{\theta}_t = \boldsymbol{\theta}_{t-1} - \eta \mathbf{V}_t$$

Where:
- $\mathbf{V}_t$: Accumulated parameter velocity ($[\mathbf{V}_{\mathbf{w}}^T, V_b]^T$), initialized to $\mathbf{0}$.
- $\beta \in [0, 1)$: Momentum damping factor ($0.90$).
- $(1 - \beta)$: Scale factor applied to the newly computed gradient step.
- $\eta$: Learning rate ($0.005$).
- $\boldsymbol{\theta}_t$: Parameter vector ($[\mathbf{w}^T, b]^T$).

```python
# Pure NumPy Implementation in app.py
dL_dw = (2.0 / m) * np.dot(X.T, residuals)
dL_db = float((2.0 / m) * np.sum(residuals))

V_w = V_w + (1.0 - beta) * dL_dw
V_b = V_b + (1.0 - beta) * dL_db

w -= lr * V_w
b -= lr * V_b
```

---

### E. Feature Contribution Attribution
The prediction equation can be partitioned into explicit factor contributions:
$$\hat{y} = b + \sum_{j=1}^n C_j, \quad C_j = w_j z_j = w_j \left(\frac{x_j - \mu_j}{\sigma_j}\right)$$

- **Baseline ($b$)**: Predicted yield when all features are at their historical average ($z_j = 0$).
- **Factor Contribution ($C_j$)**:
  - $C_j > 0$: Feature $j$ increases yield above the baseline by $C_j\text{ ton/ha}$.
  - $C_j < 0$: Feature $j$ reduces yield below the baseline by $|C_j|\text{ ton/ha}$.
- **Relative Share**: $P_j = \frac{|C_j|}{\sum_{k=1}^n |C_k|} \times 100\%$

---

## 3. Model Evaluation Metrics Suite

Evaluated on an out-of-sample test split ($20\%$ hold-out):

1. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{m} \sum_{i=1}^m (y_i - \hat{y}_i)^2$$
2. **Root Mean Squared Error (RMSE)**:
   $$\text{RMSE} = \sqrt{\text{MSE}}$$
3. **Mean Absolute Error (MAE)**:
   $$\text{MAE} = \frac{1}{m} \sum_{i=1}^m |y_i - \hat{y}_i|$$
4. **Coefficient of Determination ($R^2$ Score)**:
   $$R^2 = 1 - \frac{\sum_{i=1}^m (y_i - \hat{y}_i)^2}{\sum_{i=1}^m (y_i - \bar{y})^2}$$

---

## 4. Using Your Own CSV Dataset

The platform is designed to load and train on **your own CSV dataset**:

### Option 1: Live Web UI Uploader
1. Open the application in your browser (`http://127.0.0.1:8000`).
2. Click **"Manage / Upload CSV"** in the top navigation bar.
3. Drag and drop your `.csv` file.
4. Click **"Upload CSV & Train Models"**.
5. The application automatically detects columns, cleans missing values, fits new standard scalers, retrains both models, and refreshes all metrics instantly!

### Option 2: Direct File Placement
1. Place your CSV file inside the `data/` directory (e.g. `data/crop_data.csv`).
2. Ensure column names exist. The system automatically detects standard aliases:
   - **Rainfall**: `Rainfall_mm`, `rainfall`, `rain`, `precipitation`
   - **Temperature**: `Temperature_C`, `temperature`, `temp`
   - **Soil pH**: `Soil_pH`, `ph`, `pH`
   - **Humidity**: `Humidity_pct`, `humidity`
   - **Fertilizer**: `Fertilizer_Used_kg`, `fertilizer`, `fertilizer_kg`
   - **Target**: `Yield_ton_per_ha`, `crop_yield`, `yield`
3. If your CSV has custom column names, configure them in `dataset_config.json`:
   ```json
   {
     "dataset_path": "data/my_custom_crop_data.csv",
     "target_column": "yield",
     "feature_columns": ["rain", "temp", "ph", "hum", "fert"]
   }
   ```

---

## 5. Local Setup & Execution

### Prerequisites
- Python 3.10+ (Python 3.11 recommended)
- `pip`

### Installation
```bash
# Clone the repository
git clone https://github.com/<your-username>/<your-repo-name>.git
cd crop_yield_momentum

# Install dependencies
pip install -r requirements.txt
```

### Launching the Application
```bash
# Option A: One-click launcher with automatic browser opening
python run.py

# Option B: Direct Uvicorn command
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```
Open **`http://127.0.0.1:8000`** in your browser.

---

## 6. Deploying to Render (Free Tier)

This application is ready for deployment on **Render**:

1. **Push your code to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Deploy Crop Yield Predictor"
   git remote add origin https://github.com/<your-username>/crop_yield_predictor.git
   git branch -M main
   git push -u origin main
   ```
2. **Create Web Service on Render**:
   - Go to [dashboard.render.com](https://dashboard.render.com/).
   - Click **"New +"** $\to$ **"Web Service"**.
   - Select your GitHub repository.
3. **Configure Service Settings**:
   - **Name**: `crop-yield-predictor`
   - **Region**: Oregon (US West) or closest region
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: `Free`
4. **Click "Deploy Web Service"**.
   - Render will build the environment, install packages, load your CSV, train the model, and provide a public URL (e.g., `https://crop-yield-predictor.onrender.com`).
