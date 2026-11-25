"""
================================================================================
CROP YIELD PREDICTION & MOMENTUM OPTIMIZATION PLATFORM
Backend Service: FastAPI + Pure NumPy Vector Calculus Engine
File: app.py

PROBLEM STATEMENT:
"To develop and evaluate an interpretable crop yield prediction model and
determine whether momentum optimization trains it faster and more accurately
than standard gradient descent."
================================================================================
"""

import os
import io
import json
import base64
import logging
import shutil
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AgronomicOptimizer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(BASE_DIR, "dataset_config.json")

os.makedirs(DATA_DIR, exist_ok=True)

PROBLEM_STATEMENT = (
    "To develop and evaluate an interpretable crop yield prediction model and "
    "determine whether momentum optimization trains it faster and more accurately "
    "than standard gradient descent."
)

# Standard default feature aliases for automatic detection
FEATURE_ALIASES = {
    "rainfall": ["rainfall_mm", "rainfall", "rain", "annual_rainfall", "precipitation"],
    "temperature": ["temperature_c", "temperature", "temp", "mean_temperature", "temp_c"],
    "soil_ph": ["soil_ph", "ph", "soil_acidity", "ph_level"],
    "humidity": ["humidity_pct", "humidity", "relative_humidity", "hum"],
    "fertilizer": ["fertilizer_used_kg", "fertilizer", "fertilizer_kg", "fertilizer_usage", "fertilizer_application"]
}
TARGET_ALIASES = ["yield_ton_per_ha", "crop_yield", "yield", "production_ton", "yield_kg_per_ha", "yield_tons"]

DEFAULT_TEST_SIZE = 0.20
DEFAULT_RANDOM_STATE = 42
DEFAULT_EPOCHS = 300
DEFAULT_LEARNING_RATE = 0.005
DEFAULT_MOMENTUM_FACTOR = 0.90


def load_dataset_config() -> Dict[str, Any]:
    """Load configuration from dataset_config.json or environment."""
    config = {
        "dataset_path": os.getenv("DATASET_PATH", os.path.join(DATA_DIR, "crop_data.csv")),
        "target_column": "Yield_ton_per_ha",
        "feature_columns": [
            "Rainfall_mm",
            "Temperature_C",
            "Soil_pH",
            "Humidity_pct",
            "Fertilizer_Used_kg"
        ]
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                config.update(saved)
        except Exception as e:
            logger.warning(f"Could not read dataset_config.json: {e}")
    return config


def save_dataset_config(config: Dict[str, Any]):
    """Persist dataset configuration to dataset_config.json."""
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save dataset_config.json: {e}")


def resolve_dataset_path(preferred_path: str) -> Tuple[str, bool]:
    """Find the dataset file on disk across various candidate locations."""
    search_candidates = [
        preferred_path,
        os.path.join(BASE_DIR, preferred_path) if not os.path.isabs(preferred_path) else preferred_path,
        os.path.join(DATA_DIR, "crop_data.csv"),
        os.path.join(DATA_DIR, "crop_data"),
        os.path.join(BASE_DIR, "crop_data.csv"),
        os.path.join(BASE_DIR, "crop_data")
    ]
    for path in search_candidates:
        if path and os.path.exists(path) and os.path.isfile(path):
            return os.path.abspath(path), True
    return os.path.abspath(search_candidates[0]), False


def detect_columns(df: pd.DataFrame, configured_target: str, configured_features: List[str]) -> Tuple[str, List[str]]:
    """
    Intelligently map or detect target and input feature columns from the dataframe.
    """
    df_cols_lower = {col.lower().strip(): col for col in df.columns}

    # 1. Target column detection
    target_col = None
    if configured_target and configured_target in df.columns:
        target_col = configured_target
    elif configured_target and configured_target.lower().strip() in df_cols_lower:
        target_col = df_cols_lower[configured_target.lower().strip()]
    else:
        for alias in TARGET_ALIASES:
            if alias in df_cols_lower:
                target_col = df_cols_lower[alias]
                break
        if not target_col:
            # Fallback: pick last numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = numeric_cols[-1]

    # 2. Feature columns detection
    feature_cols = []
    if configured_features:
        for fc in configured_features:
            if fc in df.columns:
                feature_cols.append(fc)
            elif fc.lower().strip() in df_cols_lower:
                feature_cols.append(df_cols_lower[fc.lower().strip()])

    if len(feature_cols) == 0:
        # Try alias detection
        for _, aliases in FEATURE_ALIASES.items():
            for alias in aliases:
                if alias in df_cols_lower and df_cols_lower[alias] != target_col:
                    col_name = df_cols_lower[alias]
                    if col_name not in feature_cols:
                        feature_cols.append(col_name)
                    break

    # If still empty or insufficient, use all numeric columns except target
    if len(feature_cols) == 0:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c != target_col]

    return target_col, feature_cols


class ModelTelemetry:
    def __init__(self):
        self.problem_statement: str = PROBLEM_STATEMENT
        self.is_demo_mode: bool = False
        self.dataset_path: str = ""
        self.dataset_name: str = ""
        self.dataset_rows: int = 0
        self.columns_detected: List[str] = []
        
        self.feature_columns: List[str] = []
        self.target_column: str = ""
        
        self.mu_train: np.ndarray = np.array([])
        self.sigma_train: np.ndarray = np.array([])
        
        self.epochs: int = DEFAULT_EPOCHS
        self.learning_rate: float = DEFAULT_LEARNING_RATE
        self.beta: float = DEFAULT_MOMENTUM_FACTOR
        
        self.w_gd: np.ndarray = np.array([])
        self.b_gd: float = 0.0
        self.train_loss_gd: List[float] = []
        self.metrics_train_gd: Dict[str, float] = {}
        self.metrics_test_gd: Dict[str, float] = {}
        
        self.w_mom: np.ndarray = np.array([])
        self.b_mom: float = 0.0
        self.velocity_w: np.ndarray = np.array([])
        self.velocity_b: float = 0.0
        self.train_loss_mom: List[float] = []
        self.metrics_train_mom: Dict[str, float] = {}
        self.metrics_test_mom: Dict[str, float] = {}
        
        self.epochs_to_threshold_gd: Optional[int] = None
        self.epochs_to_threshold_mom: Optional[int] = None
        self.is_faster: bool = True
        self.is_more_accurate: bool = True
        self.speedup_factor: float = 1.0


state = ModelTelemetry()


def compute_loss_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error (MSE) objective function: L = (1/n) * sum((y_hat - y)^2)."""
    return float(np.mean((y_pred - y_true) ** 2))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes comprehensive evaluation metrics:
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Coefficient of Determination (R² Score)
    """
    residuals = y_pred - y_true
    mse = float(np.mean(residuals ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    ss_res = float(np.sum(residuals ** 2))
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        "mse": round(mse, 4),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }


def compute_partial_derivatives(
    X: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Explicit NumPy implementation of partial derivatives:
    dL/dw_j = (2/n) * sum((y_hat - y) * x_ij)  =>  (2/n) * X^T (y_hat - y)
    dL/db   = (2/n) * sum(y_hat - y)
    """
    m = X.shape[0]
    residuals = y_pred - y_true
    dL_dw = (2.0 / m) * np.dot(X.T, residuals)
    dL_db = float((2.0 / m) * np.sum(residuals))
    return dL_dw, dL_db


def run_vanilla_gradient_descent(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float
) -> Tuple[np.ndarray, float, List[float]]:
    """Standard Vanilla Gradient Descent: theta_t = theta_{t-1} - lr * grad(L)."""
    m, n = X_train.shape
    w = np.zeros(n, dtype=np.float64)
    b = 0.0
    history = []

    for _ in range(epochs):
        y_pred = np.dot(X_train, w) + b
        loss = compute_loss_mse(y_train, y_pred)
        history.append(loss)

        dL_dw, dL_db = compute_partial_derivatives(X_train, y_train, y_pred)
        w -= lr * dL_dw
        b -= lr * dL_db

    return w, b, history


def run_momentum_optimization(
    X_train: np.ndarray, y_train: np.ndarray, epochs: int, lr: float, beta: float
) -> Tuple[np.ndarray, float, List[float], np.ndarray, float]:
    """
    Momentum Optimization using exact specified recurrence relations:
    V_t = V_{t-1} + (1 - beta) * grad(L)
    theta_t = theta_{t-1} - lr * V_t
    """
    m, n = X_train.shape
    w = np.zeros(n, dtype=np.float64)
    b = 0.0
    V_w = np.zeros(n, dtype=np.float64)
    V_b = 0.0
    history = []

    for _ in range(epochs):
        y_pred = np.dot(X_train, w) + b
        loss = compute_loss_mse(y_train, y_pred)
        history.append(loss)

        dL_dw, dL_db = compute_partial_derivatives(X_train, y_train, y_pred)
        
        # Exact requested update equations
        V_w = V_w + (1.0 - beta) * dL_dw
        V_b = V_b + (1.0 - beta) * dL_db
        
        w -= lr * V_w
        b -= lr * V_b

    return w, b, history, V_w, V_b


def fallback_synthetic_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fallback if no dataset file exists on disk, with explicit notice."""
    np.random.seed(DEFAULT_RANDOM_STATE)
    m = 200
    features = ["Rainfall_mm", "Temperature_C", "Soil_pH", "Humidity_pct", "Fertilizer_Used_kg"]
    state.feature_columns = features
    state.target_column = "Yield_ton_per_ha"
    state.dataset_rows = m
    state.columns_detected = features + [state.target_column]
    state.dataset_name = "Synthetic Placeholder (Please upload your CSV)"

    X_syn = np.random.randn(m, len(features))
    true_weights = np.array([0.75, -0.40, 0.60, 0.25, 0.85])
    true_bias = 4.20
    y_syn = np.dot(X_syn, true_weights) + true_bias + np.random.randn(m) * 0.25

    state.mu_train = np.zeros(len(features))
    state.sigma_train = np.ones(len(features))

    split = int(m * (1.0 - DEFAULT_TEST_SIZE))
    return X_syn[:split], y_syn[:split], X_syn[split:], y_syn[split:]


def load_and_prepare_dataset() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads user CSV, identifies target and feature columns, handles missing values,
    and applies standard scaling without data leakage.
    """
    global state
    config = load_dataset_config()
    resolved_path, exists = resolve_dataset_path(config.get("dataset_path", ""))
    state.dataset_path = resolved_path
    state.dataset_name = os.path.basename(resolved_path)

    if not exists:
        logger.warning(f"Target CSV file not found at '{resolved_path}'. Enabling Demo Mode.")
        state.is_demo_mode = True
        return fallback_synthetic_data()

    try:
        df = pd.read_csv(resolved_path)
        df.columns = df.columns.str.strip()
        state.columns_detected = list(df.columns)

        target_col, feature_cols = detect_columns(
            df,
            config.get("target_column", ""),
            config.get("feature_columns", [])
        )

        if not target_col or not feature_cols:
            raise ValueError(f"Could not identify target or feature columns in {resolved_path}")

        state.target_column = target_col
        state.feature_columns = feature_cols

        # Data Cleaning: drop rows where target is NaN
        df = df.dropna(subset=[target_col])
        
        # Ensure numeric conversion
        for col in feature_cols + [target_col]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Impute missing feature values with median or drop
        for col in feature_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())

        df = df.dropna(subset=feature_cols + [target_col])
        state.dataset_rows = len(df)
        state.is_demo_mode = False
        logger.info(
            f"Successfully loaded user CSV: {state.dataset_rows} rows from '{resolved_path}'. "
            f"Target: '{target_col}', Features: {feature_cols}"
        )

        X = df[feature_cols].to_numpy(dtype=np.float64)
        y = df[target_col].to_numpy(dtype=np.float64)

        # Train/Test Split
        np.random.seed(DEFAULT_RANDOM_STATE)
        permutation = np.random.permutation(len(X))
        split_point = int(len(X) * (1.0 - DEFAULT_TEST_SIZE))
        
        train_indices = permutation[:split_point]
        test_indices = permutation[split_point:]

        X_train_raw = X[train_indices]
        y_train = y[train_indices]
        X_test_raw = X[test_indices]
        y_test = y[test_indices]

        # Feature Scaling: Mean and Std computed solely on X_train to prevent leakage
        state.mu_train = np.mean(X_train_raw, axis=0)
        state.sigma_train = np.std(X_train_raw, axis=0)
        state.sigma_train[state.sigma_train == 0.0] = 1.0

        X_train_scaled = (X_train_raw - state.mu_train) / state.sigma_train
        X_test_scaled = (X_test_raw - state.mu_train) / state.sigma_train

        return X_train_scaled, y_train, X_test_scaled, y_test

    except Exception as exc:
        logger.error(f"Failed to parse CSV dataset: {exc}. Enabling fallback placeholder.")
        state.is_demo_mode = True
        return fallback_synthetic_data()


def render_loss_plot_base64() -> str:
    """Generates empirical loss trajectory visualization comparing GD and Momentum."""
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
    fig.patch.set_facecolor("#070D0A")
    ax.set_facecolor("#0E1712")

    epochs_range = list(range(len(state.train_loss_gd)))
    ax.plot(epochs_range, state.train_loss_gd, label="Vanilla Gradient Descent", color="#F87171", lw=2.2)
    ax.plot(
        epochs_range,
        state.train_loss_mom,
        label=f"Momentum Optimization (β = {state.beta})",
        color="#34D399",
        lw=2.4
    )

    ax.set_title(
        "Empirical Loss Trajectory: Vanilla GD vs. Momentum Optimization",
        fontsize=12,
        fontweight=600,
        color="#F1F5F9",
        pad=12
    )
    ax.set_xlabel("Optimization Epochs", fontsize=10, color="#94A3B8")
    ax.set_ylabel("Mean Squared Error (tons/ha)²", fontsize=10, color="#94A3B8")
    ax.grid(True, linestyle=":", alpha=0.25, color="#34D399")
    
    legend = ax.legend(frameon=True, facecolor="#070D0A", edgecolor="#34D399", fontsize=9)
    for text in legend.get_texts():
        text.set_color("#F1F5F9")

    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def execute_model_training(epochs: int, lr: float, beta: float):
    """Executes full training pipeline, computes metrics, and updates global state."""
    global state
    state.epochs = epochs
    state.learning_rate = lr
    state.beta = beta

    X_train, y_train, X_test, y_test = load_and_prepare_dataset()

    # Train Vanilla Gradient Descent
    w_gd, b_gd, loss_gd = run_vanilla_gradient_descent(X_train, y_train, epochs, lr)
    state.w_gd, state.b_gd, state.train_loss_gd = w_gd, b_gd, loss_gd

    # Train Momentum Optimization
    w_mom, b_mom, loss_mom, v_w, v_b = run_momentum_optimization(X_train, y_train, epochs, lr, beta)
    state.w_mom, state.b_mom, state.velocity_w, state.velocity_b, state.train_loss_mom = w_mom, b_mom, v_w, v_b, loss_mom

    # Evaluate on Train and Test Sets
    pred_train_gd = np.dot(X_train, w_gd) + b_gd
    pred_test_gd = np.dot(X_test, w_gd) + b_gd
    state.metrics_train_gd = compute_metrics(y_train, pred_train_gd)
    state.metrics_test_gd = compute_metrics(y_test, pred_test_gd)

    pred_train_mom = np.dot(X_train, w_mom) + b_mom
    pred_test_mom = np.dot(X_test, w_mom) + b_mom
    state.metrics_train_mom = compute_metrics(y_train, pred_train_mom)
    state.metrics_test_mom = compute_metrics(y_test, pred_test_mom)

    # Benchmark Speedup to Target Error Threshold
    target_thresh = min(loss_gd[-1], loss_mom[-1]) * 1.25
    state.epochs_to_threshold_gd = next((i for i, v in enumerate(loss_gd) if v <= target_thresh), epochs)
    state.epochs_to_threshold_mom = next((i for i, v in enumerate(loss_mom) if v <= target_thresh), epochs)
    
    if state.epochs_to_threshold_mom and state.epochs_to_threshold_mom > 0:
        state.speedup_factor = round(state.epochs_to_threshold_gd / state.epochs_to_threshold_mom, 2)
    else:
        state.speedup_factor = 1.0

    state.is_faster = (state.epochs_to_threshold_mom <= state.epochs_to_threshold_gd)
    state.is_more_accurate = (state.metrics_test_mom["mse"] <= state.metrics_test_gd["mse"])


# Initialize and train models on module load
try:
    execute_model_training(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM_FACTOR)
except Exception as e:
    logger.warning(f"Initial model training on import deferred: {e}")


app = FastAPI(title="Crop Yield Predictor & Agronomic Momentum Engine")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)


class RetrainPayload(BaseModel):
    epochs: int = Field(DEFAULT_EPOCHS, ge=20, le=10000)
    learning_rate: float = Field(DEFAULT_LEARNING_RATE, ge=0.0001, le=0.5)
    momentum_factor: float = Field(DEFAULT_MOMENTUM_FACTOR, ge=0.0, le=0.99)


class MathCalculatorPayload(BaseModel):
    rainfall: float
    temperature: float
    soil_ph: float
    humidity: float
    fertilizer: float
    actual_yield: float


class ConfigureDatasetPayload(BaseModel):
    target_column: str
    feature_columns: List[str]


@app.on_event("startup")
def app_startup():
    execute_model_training(DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_MOMENTUM_FACTOR)


@app.get("/", response_class=HTMLResponse)
async def serve_index(request: Request):
    plot_b64 = render_loss_plot_base64()
    weights_summary = [
        {
            "feature": col,
            "mean": round(float(state.mu_train[i]), 2) if i < len(state.mu_train) else 0.0,
            "std": round(float(state.sigma_train[i]), 2) if i < len(state.sigma_train) else 1.0,
            "weight_gd": round(float(state.w_gd[i]), 4) if i < len(state.w_gd) else 0.0,
            "weight_mom": round(float(state.w_mom[i]), 4) if i < len(state.w_mom) else 0.0,
            "velocity": round(float(state.velocity_w[i]), 4) if i < len(state.velocity_w) else 0.0
        }
        for i, col in enumerate(state.feature_columns)
    ]
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "problem_statement": state.problem_statement,
            "is_demo_mode": state.is_demo_mode,
            "dataset_rows": state.dataset_rows,
            "dataset_path": state.dataset_path,
            "dataset_name": state.dataset_name,
            "feature_columns": state.feature_columns,
            "target_column": state.target_column,
            "columns_detected": state.columns_detected,
            "epochs": state.epochs,
            "learning_rate": state.learning_rate,
            "beta": state.beta,
            "bias_gd": round(state.b_gd, 4),
            "bias_mom": round(state.b_mom, 4),
            "metrics_gd": state.metrics_test_gd,
            "metrics_mom": state.metrics_test_mom,
            "train_mse_gd": state.metrics_train_gd.get("mse", 0.0),
            "train_mse_mom": state.metrics_train_mom.get("mse", 0.0),
            "test_mse_gd": state.metrics_test_gd.get("mse", 0.0),
            "test_mse_mom": state.metrics_test_mom.get("mse", 0.0),
            "test_rmse_gd": state.metrics_test_gd.get("rmse", 0.0),
            "test_rmse_mom": state.metrics_test_mom.get("rmse", 0.0),
            "test_mae_gd": state.metrics_test_gd.get("mae", 0.0),
            "test_mae_mom": state.metrics_test_mom.get("mae", 0.0),
            "test_r2_gd": state.metrics_test_gd.get("r2", 0.0),
            "test_r2_mom": state.metrics_test_mom.get("r2", 0.0),
            "epochs_to_threshold_gd": state.epochs_to_threshold_gd,
            "epochs_to_threshold_mom": state.epochs_to_threshold_mom,
            "speedup_factor": state.speedup_factor,
            "is_faster": state.is_faster,
            "is_more_accurate": state.is_more_accurate,
            "weights_summary": weights_summary,
            "plot_b64": plot_b64
        }
    )


@app.post("/api/predict")
async def api_predict(request: Request):
    """
    Accepts arbitrary feature inputs, computes predicted crop yield,
    and returns detailed feature contribution breakdown:
    y_hat = b + sum(w_j * z_j).
    """
    payload = await request.json()
    
    # Extract values in order of state.feature_columns
    raw_values = []
    for col in state.feature_columns:
        if col in payload:
            raw_values.append(float(payload[col]))
        else:
            # Fallback to mean if missing in payload
            idx = state.feature_columns.index(col)
            raw_values.append(float(state.mu_train[idx]))

    raw_input = np.array(raw_values, dtype=np.float64)
    scaled_input = (raw_input - state.mu_train) / state.sigma_train

    # Linear Regression predictions
    pred_gd = float(np.dot(scaled_input, state.w_gd) + state.b_gd)
    pred_mom = float(np.dot(scaled_input, state.w_mom) + state.b_mom)

    # Feature Contribution calculation: C_j = w_j * z_j
    contributions = scaled_input * state.w_mom
    total_abs_contrib = float(np.sum(np.abs(contributions)))
    if total_abs_contrib == 0.0:
        total_abs_contrib = 1.0

    feature_breakdown = []
    for i, col in enumerate(state.feature_columns):
        c_val = float(contributions[i])
        pct = round((abs(c_val) / total_abs_contrib) * 100.0, 1)
        feature_breakdown.append({
            "feature": col,
            "raw_value": round(float(raw_input[i]), 2),
            "mean": round(float(state.mu_train[i]), 2),
            "std": round(float(state.sigma_train[i]), 2),
            "z_score": round(float(scaled_input[i]), 3),
            "weight": round(float(state.w_mom[i]), 4),
            "contribution_tons": round(c_val, 4),
            "percentage": pct,
            "direction": "positive" if c_val >= 0 else "negative"
        })

    return {
        "prediction_momentum": round(max(0.0, pred_mom), 2),
        "prediction_vanilla_gd": round(max(0.0, pred_gd), 2),
        "baseline_yield": round(state.b_mom, 2),
        "unit": "ton/ha",
        "raw_scaled_features": [round(float(z), 3) for z in scaled_input],
        "feature_contributions": feature_breakdown,
        "is_demo_mode": state.is_demo_mode
    }


@app.post("/api/retrain")
async def api_retrain(payload: RetrainPayload):
    """Retrain model with updated hyperparameters and return new metrics."""
    execute_model_training(payload.epochs, payload.learning_rate, payload.momentum_factor)
    return {
        "status": "success",
        "metrics_gd": state.metrics_test_gd,
        "metrics_mom": state.metrics_test_mom,
        "train_mse_gd": state.metrics_train_gd.get("mse", 0.0),
        "train_mse_mom": state.metrics_train_mom.get("mse", 0.0),
        "test_mse_gd": state.metrics_test_gd.get("mse", 0.0),
        "test_mse_mom": state.metrics_test_mom.get("mse", 0.0),
        "test_rmse_gd": state.metrics_test_gd.get("rmse", 0.0),
        "test_rmse_mom": state.metrics_test_mom.get("rmse", 0.0),
        "test_mae_gd": state.metrics_test_gd.get("mae", 0.0),
        "test_mae_mom": state.metrics_test_mom.get("mae", 0.0),
        "test_r2_gd": state.metrics_test_gd.get("r2", 0.0),
        "test_r2_mom": state.metrics_test_mom.get("r2", 0.0),
        "epochs_to_threshold_gd": state.epochs_to_threshold_gd,
        "epochs_to_threshold_mom": state.epochs_to_threshold_mom,
        "speedup_factor": state.speedup_factor,
        "is_faster": state.is_faster,
        "is_more_accurate": state.is_more_accurate,
        "plot_b64": render_loss_plot_base64()
    }


@app.post("/api/calculate_math")
async def api_calculate_math(payload: MathCalculatorPayload):
    """Live calculus sandbox endpoint: step-by-step differentiation and momentum update."""
    raw_vec = np.array([
        payload.rainfall,
        payload.temperature,
        payload.soil_ph,
        payload.humidity,
        payload.fertilizer
    ], dtype=np.float64)

    # Use first 5 feature scalers if available
    n = min(5, len(state.mu_train))
    mu_subset = state.mu_train[:n]
    sigma_subset = state.sigma_train[:n]
    w_subset = state.w_mom[:n]
    v_subset = state.velocity_w[:n]

    scaled_vec = (raw_vec[:n] - mu_subset) / sigma_subset
    y_hat = float(np.dot(scaled_vec, w_subset) + state.b_mom)
    residual = float(y_hat - payload.actual_yield)
    
    # Analytical partial derivatives
    sample_grad_w = 2.0 * residual * scaled_vec
    sample_grad_b = 2.0 * residual
    
    # Exact Momentum update simulation
    simulated_V_w = v_subset + (1.0 - state.beta) * sample_grad_w
    simulated_w_update = w_subset - state.learning_rate * simulated_V_w

    return {
        "scaled_features": [round(float(v), 4) for v in scaled_vec],
        "predicted_yield": round(y_hat, 4),
        "residual_error": round(residual, 4),
        "squared_error": round(residual ** 2, 4),
        "partial_derivatives_w": [round(float(g), 4) for g in sample_grad_w],
        "partial_derivative_b": round(sample_grad_b, 4),
        "simulated_velocity_w": [round(float(v), 4) for v in simulated_V_w],
        "simulated_weights_next": [round(float(w), 4) for w in simulated_w_update]
    }


@app.post("/api/dataset/upload")
async def api_upload_dataset(file: UploadFile = File(...)):
    """
    Allows user to upload their own CSV file directly.
    Saves the file, inspects columns, and retrains the model on the uploaded dataset.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files (.csv) are supported.")

    dest_filename = os.path.basename(file.filename)
    dest_path = os.path.join(DATA_DIR, dest_filename)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Update config to point to new dataset
    config = load_dataset_config()
    config["dataset_path"] = os.path.relpath(dest_path, BASE_DIR)
    save_dataset_config(config)

    # Retrain on user's newly uploaded dataset
    execute_model_training(state.epochs, state.learning_rate, state.beta)

    return {
        "status": "success",
        "message": f"Successfully loaded user dataset '{dest_filename}' with {state.dataset_rows} rows.",
        "filename": dest_filename,
        "rows": state.dataset_rows,
        "target_column": state.target_column,
        "feature_columns": state.feature_columns,
        "columns_detected": state.columns_detected,
        "metrics_mom": state.metrics_test_mom,
        "metrics_gd": state.metrics_test_gd
    }


@app.post("/api/dataset/configure")
async def api_configure_dataset(payload: ConfigureDatasetPayload):
    """Configures custom target and feature columns and retrains."""
    config = load_dataset_config()
    config["target_column"] = payload.target_column
    config["feature_columns"] = payload.feature_columns
    save_dataset_config(config)

    execute_model_training(state.epochs, state.learning_rate, state.beta)
    return {
        "status": "success",
        "target_column": state.target_column,
        "feature_columns": state.feature_columns,
        "metrics_mom": state.metrics_test_mom,
        "metrics_gd": state.metrics_test_gd
    }


@app.get("/api/dataset/info")
async def api_dataset_info():
    """Returns dataset schema, sample rows, and configuration info."""
    sample_rows = []
    if os.path.exists(state.dataset_path):
        try:
            df = pd.read_csv(state.dataset_path)
            sample_rows = df.head(5).to_dict(orient="records")
        except Exception:
            pass

    return {
        "dataset_path": state.dataset_path,
        "dataset_name": state.dataset_name,
        "dataset_rows": state.dataset_rows,
        "is_demo_mode": state.is_demo_mode,
        "target_column": state.target_column,
        "feature_columns": state.feature_columns,
        "all_columns": state.columns_detected,
        "sample_preview": sample_rows
    }


@app.get("/api/status")
async def api_status():
    """Telemetry and health status endpoint."""
    return {
        "status": "healthy",
        "problem_statement": state.problem_statement,
        "demo_mode": state.is_demo_mode,
        "dataset_path": state.dataset_path,
        "dataset_name": state.dataset_name,
        "dataset_rows": state.dataset_rows,
        "columns_detected": state.columns_detected,
        "target_column": state.target_column,
        "feature_columns": state.feature_columns,
        "metrics_test_gd": state.metrics_test_gd,
        "metrics_test_mom": state.metrics_test_mom,
        "speedup_factor": state.speedup_factor,
        "is_faster": state.is_faster,
        "is_more_accurate": state.is_more_accurate,
        "epochs": state.epochs,
        "learning_rate": state.learning_rate,
        "beta": state.beta
    }
