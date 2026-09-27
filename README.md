# Time Series Forecaster

Web application for testing multiple time series forecasting methods with an interactive interface.

## Architecture

- **Frontend**: Next.js 14 with TypeScript, Tailwind CSS, and Plotly (deployed on Vercel)
- **Backend**: FastAPI with Polars, scikit-learn, XGBoost, statsmodels and Prophet (deployed on Render)

The browser calls the backend directly. The API is public (demo app): there is no API key,
abuse is limited server-side (CORS, request size limits).

## Features

- Upload CSV time series data, automatic frequency detection and data analysis
  (ACF/PACF, suggested lags, seasonality, outliers, trend)
- Configure training and prediction periods, and a forecast horizon
- Train multiple models simultaneously:
  - Lag baseline
  - Linear Regression and XGBoost with lag, temporal and exogenous features
    (target mode raw or residual, SHAP analysis for XGBoost)
  - ARIMA
  - Prophet
- Multi-step evaluation: the prediction period is split into blocks of `horizon` steps,
  forecast recursively within each block, with metrics by horizon step
- Exogenous variables are either *known ahead* (calendar, planned promotions) or not:
  in the latter case only lags >= horizon are allowed, to avoid leaking future values
- Compare model performance with interactive charts, export forecasts to CSV

## Installation & Setup

Requirements: [uv](https://docs.astral.sh/uv/) (installs Python 3.12 if needed) and Node.js 18+.

### Backend

```bash
cd backend
uv sync
uv run python main.py
```

The API will be available at `http://localhost:8000` (auto-reload on code changes).
API documentation: `http://localhost:8000/docs`

Run the tests with `uv run pytest`.

Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `ALLOWED_ORIGINS` | `http://localhost:3000,http://127.0.0.1:3000` | Frontend origins allowed by CORS, comma-separated |
| `MAX_BODY_MB` | `20` | Maximum request size |
| `MAX_ROWS` | `100000` | Maximum dataset rows |
| `MAX_MODELS` | `10` | Maximum models per training request |

### Frontend

```bash
cd frontend
cp .env.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
```

The app will be available at `http://localhost:3000`

## Usage

1. **Upload Data**: Drag and drop or select a CSV file with time series data
   (examples in [`demo_data/`](demo_data/))
2. **Configure Models**: Select models from the library and configure their parameters
3. **Set Validation Strategy**: Define training and prediction periods and the forecast horizon
4. **Train & Compare**: Launch training and view results with metrics and visualizations
