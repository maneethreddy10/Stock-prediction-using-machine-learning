import streamlit as st
import pandas as pd
import yfinance as yf
import io
import requests
import psycopg2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# Try importing Statsmodels for ARIMA
try:
    from statsmodels.tsa.arima.model import ARIMA

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# -----------------------------------------------------------------------------
# DATABASE CONFIGURATION
# -----------------------------------------------------------------------------
DB_NAME = "stocker"
DB_USER = "postgres"
DB_PASSWORD = "root"
DB_HOST = "localhost"


# -----------------------------------------------------------------------------
# DATABASE FUNCTIONS
# -----------------------------------------------------------------------------
def connect_to_db():
    try:
        return psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
        )
    except Exception as e:
        st.error(f"DB connection error: {e}")
        return None


def delete_old_model_results(symbol):
    conn = connect_to_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM model_results WHERE stock_symbol = %s;", (symbol,))
        conn.commit()
    except Exception as e:
        st.error(f"Failed to delete old model results: {e}")
    finally:
        conn.close()


def safe_float(x):
    try:
        if x is None:
            return None
        xf = float(x)
        return xf if not (np.isnan(xf) or np.isinf(xf)) else None
    except:
        return None


def save_full_model_result(symbol, stock_index, model_name, metrics):
    conn = connect_to_db()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            query = """
            INSERT INTO model_results (
                stock_symbol, stock_index, model_name,
                rmse_1_week, rmse_1_month, rmse_1_year,
                mape_1_week, mape_1_month, mape_1_year
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s);
            """
            horizon_keys = ['1_week', '1_month', '1_year']
            rmse_vals = [safe_float(metrics[h][0]) for h in horizon_keys]
            mape_vals = [safe_float(metrics[h][1]) for h in horizon_keys]
            params = [symbol, stock_index, model_name] + rmse_vals + mape_vals
            cur.execute(query, params)
        conn.commit()
    except Exception as e:
        st.error(f"Error saving model results: {e}")
    finally:
        conn.close()


def fetch_model_results(symbol):
    conn = connect_to_db()
    if not conn:
        return pd.DataFrame()
    try:
        query = """SELECT model_name,
        rmse_1_week, rmse_1_month, rmse_1_year,
        mape_1_week, mape_1_month, mape_1_year
        FROM model_results WHERE stock_symbol = %s;"""
        df = pd.read_sql(query, conn, params=(symbol,))
        return df
    except:
        return pd.DataFrame()
    finally:
        conn.close()


# -----------------------------------------------------------------------------
# CUSTOM CLASSES (ARIMA WRAPPER)
# -----------------------------------------------------------------------------
class SklearnARIMA:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model_res = None
        self.train_data = None

    def fit(self, X, y):
        self.train_data = y
        if STATSMODELS_AVAILABLE:
            try:
                y_reset = y.reset_index(drop=True)
                model = ARIMA(y_reset, order=self.order)
                self.model_res = model.fit()
            except Exception as e:
                pass
        return self

    def predict(self, X):
        if self.model_res is None:
            return np.zeros(len(X))
        n_periods = len(X)
        start = len(self.train_data)
        end = start + n_periods - 1
        try:
            pred = self.model_res.predict(start=start, end=end)
            return pred.values
        except Exception as e:
            return np.zeros(n_periods)


# -----------------------------------------------------------------------------
# METRICS & FEATURES
# -----------------------------------------------------------------------------
def compute_regression_metrics(y_true, y_pred):
    if len(y_true) < 2 or len(y_pred) < 2:
        return np.nan, np.nan
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return rmse, mape


def add_all_features(df):
    df = df.copy()
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['weekofyear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['month'] = df['Date'].dt.month

    # Unconditionally calculate return to ensure new appended rows get calculated
    df['Return'] = df['Close'].pct_change()

    # Lags
    for i in range(1, 6):
        df[f'lag_return_{i}'] = df['Return'].shift(i)

    # Rolling stats
    sma_7 = df['Close'].rolling(7).mean()
    df['dist_from_sma_7'] = (df['Close'] - sma_7) / sma_7

    df['rolling_std_7_pct'] = df['Return'].rolling(7).std()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    exp12 = df['Close'].ewm(span=12, adjust=False).mean()
    exp26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp12 - exp26
    df['MACD_pct'] = macd / df['Close']
    df['MACD_signal_pct'] = df['MACD_pct'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_width_pct'] = (4 * std20) / sma20

    return df


@st.cache_data(show_spinner=False)
def load_all_nse_symbols():
    url = "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        symbols = df['SYMBOL'].str.strip().tolist()
        return [s + ".NS" for s in symbols]
    except Exception as e:
        st.error(f"Failed to load NSE symbols: {e}")
        return []


@st.cache_data(show_spinner=False)
def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, period='5y', interval='1d', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# PREPARE FEATURES FOR RETURN PREDICTION
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def prepare_features(data):
    df = data[['Date', 'Close']].copy()

    # 1. Add stationary features
    df = add_all_features(df)

    # 2. Shift Return to create Target
    df['Target_Return'] = df['Return'].shift(-1)

    feature_cols = [c for c in df.columns if c not in ['Date', 'Close', 'Return', 'Target_Return']]

    df.dropna(inplace=True)

    X = df[feature_cols]
    y = df['Target_Return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    test_indices = X_test.index
    test_base_prices = df.loc[test_indices, 'Close']

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, test_base_prices, scaler


# -----------------------------------------------------------------------------
# EVALUATION
# -----------------------------------------------------------------------------
def evaluate_on_test_horizons(model, X_test, y_test_returns, test_base_prices, horizons):
    metrics = {}
    try:
        if isinstance(model, SklearnARIMA):
            pred_returns = model.predict(X_test)
        else:
            pred_returns = model.predict(X_test.values)
    except:
        pred_returns = np.zeros(len(X_test))

    pred_prices = test_base_prices.values * (1 + pred_returns)
    actual_prices = test_base_prices.values * (1 + y_test_returns.values)

    for h, d in horizons.items():
        n = min(d, len(actual_prices))
        if n < 2:
            metrics[h] = (np.nan, np.nan)
            continue

        true_vals = actual_prices[-n:]
        pred_vals = pred_prices[-n:]
        metrics[h] = compute_regression_metrics(true_vals, pred_vals)

    return metrics, pred_prices, actual_prices


def get_best_models_per_horizon(df):
    horizons = ['1_week', '1_month', '1_year']
    best_models = {}
    for h in horizons:
        col = f"rmse_{h}"
        if col not in df.columns or df[col].isnull().all():
            continue
        idx = df[col].idxmin()
        if pd.isnull(idx):
            continue
        best_models[h] = df.at[idx, 'model_name']
    return best_models


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------
def plot_actual_vs_predicted_historical(test_dates, actual_prices, pred_prices, horizon):
    min_len = min(len(test_dates), len(actual_prices), len(pred_prices))
    results_df = pd.DataFrame({
        'Date': test_dates.iloc[-min_len:],
        'Actual': actual_prices[-min_len:],
        'Predicted': pred_prices[-min_len:]
    })

    max_date = results_df['Date'].max()

    if horizon == '1_week':
        cutoff_date = max_date - timedelta(days=14)
        title_context = "Last 14 Days"
    elif horizon == '1_month':
        cutoff_date = max_date - timedelta(days=90)
        title_context = "Last 90 Days"
    else:
        cutoff_date = max_date - timedelta(days=365)
        title_context = "Last 1 Year"

    plot_df = results_df[results_df['Date'] >= cutoff_date].copy()

    if plot_df.empty:
        st.warning(f"Not enough data to plot {horizon} history.")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(plot_df['Date'], plot_df['Actual'], label='Actual Price', color='blue', linewidth=2)
    ax.plot(plot_df['Date'], plot_df['Predicted'], label='Predicted Price', color='orange', linestyle='--', linewidth=2)
    ax.fill_between(plot_df['Date'], plot_df['Actual'], plot_df['Predicted'], color='gray', alpha=0.1)
    ax.set_title(f"Validation: Actual vs Predicted - {title_context}", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

    if horizon == '1_year':
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))

    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_metric_comparison(df, metric, best_models, bar_width=0.15):
    horizons = ['1_week', '1_month', '1_year']
    colors = {'rmse': '#ff7f0e', 'mape': '#1f77b4'}
    base_color = colors.get(metric, '#1f77b4')
    model_names = df['model_name']
    indices = list(range(len(model_names)))
    fig, ax = plt.subplots(figsize=(16, 8))
    for i, horizon in enumerate(horizons):
        col = f"{metric}_{horizon}"
        vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
        pos = [x + i * bar_width for x in indices]
        colors_list = [base_color if m == best_models.get(horizon) else 'lightgray' for m in model_names]
        bars = ax.bar(pos, vals, width=bar_width, color=colors_list, label=horizon.replace('_', ' ').title())
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h), xytext=(0, 4),
                            textcoords='offset points',
                            ha='center', va='bottom', fontsize=8)
    ax.set_xticks([r + bar_width * (len(horizons) - 1) / 2 for r in indices])
    ax.set_xticklabels(model_names, rotation=35, ha='right', fontsize=12)
    ax.set_title(f"{metric.upper()} Metric Comparison (Lower is Better)", fontsize=16)
    ax.set_ylabel('Score')
    ax.legend(title='Horizon')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# -----------------------------------------------------------------------------
# RECURSIVE PREDICTION
# -----------------------------------------------------------------------------
def predict_next_days(model, last_df, n_days, feature_cols, scaler):
    df_temp = last_df.copy()
    preds_prices, dates = [], []

    is_arima = isinstance(model, SklearnARIMA)
    if is_arima:
        dummy_X = range(n_days)
        forecast_returns = model.predict(dummy_X)

        current_price = df_temp['Close'].iloc[-1]
        last_date = df_temp['Date'].iloc[-1]
        current_date = last_date

        for ret in forecast_returns:
            new_price = current_price * (1 + ret)
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

            preds_prices.append(float(new_price))
            dates.append(current_date)
            current_price = new_price

    else:
        for _ in range(n_days):
            # Safe-guard against stray NaNs before passing to scaler
            df_temp.ffill(inplace=True)
            df_temp.bfill(inplace=True)

            last_features = df_temp[feature_cols].iloc[-1].values.reshape(1, -1)
            last_features_scaled = scaler.transform(last_features)

            pred_return = model.predict(last_features_scaled)[0]

            last_close = df_temp['Close'].iloc[-1]
            pred_price = last_close * (1 + pred_return)

            last_date = df_temp['Date'].iloc[-1]
            next_date = last_date + timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)

            preds_prices.append(float(pred_price))
            dates.append(next_date)

            new_row = pd.DataFrame({'Date': [next_date], 'Close': [pred_price]})
            df_temp = pd.concat([df_temp, new_row], ignore_index=True)

            # Recalculate indicators for the appended row to feed into the next iteration
            df_temp = add_all_features(df_temp)

    return preds_prices, dates


# -----------------------------------------------------------------------------
# PREDICTION UI COMPONENT
# -----------------------------------------------------------------------------
def display_prediction_card(model_name, current_price, predicted_price):
    change_pct = ((predicted_price - current_price) / current_price) * 100

    if change_pct >= 0:
        bg_color = "#1e3b22"
        text_color = "#4caf50"
        arrow = "↑"
    else:
        bg_color = "#3b1e1e"
        text_color = "#f44336"
        arrow = "↓"

    st.markdown(f"""
    <div style="padding: 10px 0; margin-bottom: 20px;">
        <div style="color: #e0e0e0; font-size: 16px; font-weight: 500; margin-bottom: 5px;">
            Predicted Price (via {model_name})
        </div>
        <div style="font-size: 42px; font-weight: bold; color: white; margin-bottom: 10px;">
            ₹{predicted_price:,.2f}
        </div>
        <div style="display: inline-block; background-color: {bg_color}; color: {text_color}; padding: 6px 14px; border-radius: 20px; font-size: 16px; font-weight: bold;">
            {arrow} {abs(change_pct):.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.title("Stocker - Return-Based Prediction")
    st.info("Models now reliably predict Percentage Return utilizing stationary indicators and scaled features.")

    all_symbols = load_all_nse_symbols()
    selected_symbol = st.selectbox("Select Stock Symbol", all_symbols)

    if not selected_symbol:
        st.info("Please select a stock symbol to proceed.")
        return

    delete_old_model_results(selected_symbol)

    with st.spinner("Fetching stock data..."):
        data = fetch_stock_data(selected_symbol)
    if data is None or data.empty:
        st.warning("No stock data fetched.")
        return

    st.subheader(f"Latest data for {selected_symbol}")
    st.dataframe(data.tail(10))

    if len(data) < 100:
        st.warning("At least 100 data points required.")
        return

    X_train, X_test, y_train, y_test, feature_cols, test_base_prices, scaler = prepare_features(data)

    test_indices = X_test.index
    test_dates = data.loc[test_indices, 'Date']

    rf_est = RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_split=10, random_state=42)
    xgb_est = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42,
                           verbosity=0) if XGBRegressor else None
    lr_est = LinearRegression()

    hybrid_estimators = [('rf', rf_est), ('lr', lr_est)]
    if xgb_est:
        hybrid_estimators.append(('xgb', xgb_est))

    models = {
        "XGBoost (Tuned)": xgb_est,
        "Random Forest (Tuned)": rf_est,
        "Ridge Regression": Ridge(alpha=1.0, random_state=42),
        "Linear Regression": LinearRegression(),
        "SVM": SVR(kernel='rbf'),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "Hybrid (Voting Regressor)": VotingRegressor(estimators=hybrid_estimators)
    }

    if STATSMODELS_AVAILABLE:
        models["ARIMA (Time Series)"] = SklearnARIMA(order=(5, 1, 0))

    models = {k: v for k, v in models.items() if v is not None}
    horizons = {'1_week': 5, '1_month': 22, '1_year': 252}

    st.write("Training models on Returns...")
    progress = st.progress(0)

    model_predictions = {}

    for i, (name, mdl) in enumerate(models.items()):
        try:
            if isinstance(mdl, SklearnARIMA):
                mdl.fit(X_train, y_train)
            else:
                mdl.fit(X_train.values, y_train.values)

            if not X_test.empty:
                metrics, pred_prices, actual_prices = evaluate_on_test_horizons(mdl, X_test, y_test, test_base_prices,
                                                                                horizons)
                save_full_model_result(selected_symbol, 'NSE Index', name, metrics)

                model_predictions[name] = {
                    'pred_prices': pred_prices,
                    'actual_prices': actual_prices
                }

        except Exception as e:
            st.error(f"Training failed for {name}: {e}")
        progress.progress((i + 1) / len(models))

    st.success("Model training complete.")

    model_scores_df = fetch_model_results(selected_symbol)
    if model_scores_df.empty:
        st.warning("No model performance data found.")
        return

    best_models = get_best_models_per_horizon(model_scores_df)

    st.subheader("Model Performance Comparison (RMSE on Price)")
    with st.expander("Show Metric Comparison Charts"):
        for metric in ['rmse', 'mape']:
            plot_metric_comparison(model_scores_df, metric, best_models)

    # Display Analysis
    st.subheader("Analysis: Validation & Future Forecast")

    hist_data = data[['Date', 'Close']].copy()
    hist_data_features = add_all_features(hist_data)
    hist_data_features.dropna(inplace=True)
    forecast_days_map = {'1_week': 5, '1_month': 22, '1_year': 252}

    for horizon, model_name in best_models.items():
        if model_name not in models:
            continue

        model = models[model_name]

        st.markdown("---")
        st.markdown(f"### Horizon: {horizon.replace('_', ' ').title()} (Best Model: {model_name})")

        tab1, tab2 = st.tabs(["Historical Validation (Graph)", "Future Forecast (Table)"])

        with tab1:
            if model_name in model_predictions:
                preds = model_predictions[model_name]['pred_prices']
                actuals = model_predictions[model_name]['actual_prices']
                plot_actual_vs_predicted_historical(test_dates, actuals, preds, horizon)

        with tab2:
            days = forecast_days_map[horizon]

            future_preds, future_dates = predict_next_days(model, hist_data_features, days, feature_cols, scaler)

            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})

            current_price = hist_data['Close'].iloc[-1]
            final_pred_price = future_preds[-1]
            display_prediction_card(model_name, current_price, final_pred_price)

            st.write("Predicted Values (Future):")
            st.dataframe(future_df)


if __name__ == "__main__":
    main()
