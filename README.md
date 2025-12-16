# Time Series Forecaster

Web application for testing multiple time series forecasting methods with an interactive interface.

## Architecture

- **Frontend**: Next.js 14 with TypeScript, Tailwind CSS, and Recharts
- **Backend**: FastAPI with Polars and scikit-learn

## Features

- Upload CSV time series data
- Configure training and prediction periods
- Train multiple models simultaneously:
  - Linear Regression with lag features
  - Target mode: raw or residual (differencing)
  - Feature standardization option
  - More models coming soon (ARIMA, Prophet, XGBoost, N-BEATS)
- Compare model performance with interactive charts
- Feature importance visualization

## Installation & Setup

### Backend

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a Python virtual environment:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the backend server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`  
API documentation: `http://localhost:8000/docs`

### Frontend

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

The app will be available at `http://localhost:3000`

## Usage

1. **Upload Data**: Drag and drop or select a CSV file with time series data
2. **Configure Models**: Select models from the library and configure their parameters
3. **Set Validation Strategy**: Define training and prediction periods
4. **Train & Compare**: Launch training and view results with metrics and visualizations

## Requirements

- Python 3.8+
- Node.js 18+
- npm or yarn
