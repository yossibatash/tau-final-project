# S&P 500 Stock Prediction with News Sentiment Analysis

**Master's Degree Final Project - Tel Aviv University**

A deep learning project that combines traditional technical analysis with news sentiment analysis to predict S&P 500 index movements using LSTM neural networks.

## 🎯 Project Overview

This project explores the integration of financial market data with news sentiment analysis to create more accurate stock prediction models. By combining SPY ticker data with quantified news sentiment features, the system attempts to capture both technical patterns and market sentiment to predict future price movements.

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Processing    │    │   ML Pipeline   │
│                 │    │                 │    │                 │
│ • S&P 500 Data  │───▶│ • Apache Spark  │───▶│ • LSTM Models   │
│ • News Articles │    │ • Feature Eng.  │    │ • TensorFlow    │
└─────────────────┘    │ • PCA Selection │    │ • Simulation    │
                       └─────────────────┘    └─────────────────┘
```

## 📊 Key Features

- **Multi-timeframe Analysis**: 1, 5, and 15-minute intervals.
- **Sentiment Integration**: News sentiment from 44+ top S&P 500 companies.
- **Feature Engineering**: 80+ technical and sentiment-based features.
- **LSTM Architecture**: Two-layer LSTM with 50 units each, 20-timestep lookback.
- **Trading Simulation**: Automated buy/sell/hold strategy with backtesting.
- **Scalable Processing**: Apache Spark for large-scale data processing.

## 🚀 Performance Results

| Model | Time Frame | Initial Capital | Final Value | Profit    | Return |
|-------|------------|-----------------|-------------|-----------|--------|
| LSTM  | 1-minute   | $10,000.00      | $11,043.72  | $1,043.72 | 10.44% |
| LSTM  | 5-minute   | $10,000.00      | $10,728.47  | $728.47   | 7.28%  |
| LSTM  | 15-minute  | $10,000.00      | $10,370.89  | $370.89   | 3.71%  |

## 🛠️ Installation

### Prerequisites
- Python 3.9+
- Apache Spark
- Alpha Vantage API Key

### Setup
```bash
# Clone the repository
git clone https://github.com/yossibatash/tau-final-project
cd tau-final-project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize Spark (if needed)
pip install findspark
```

### Quickstart
```bash
source venv/bin/activate
pip install -r requirements.txt
jupyter notebook  # open notebooks listed under Usage Workflow
```

### API Configuration
1. Get your free API key from [Alpha Vantage](https://www.alphavantage.co/support/#api-key)
2. Either set an environment variable before launching Jupyter:
   - macOS/Linux: `export ALPHAVANTAGE_API_KEY="<your_key>"`
   - Windows (PowerShell): `$Env:ALPHAVANTAGE_API_KEY="<your_key>"`
   Or paste the key into the first config cell of `Extract_SnP_500_index_data.ipynb` and `Extract_News_Sentiment_data.ipynb`.
3. Alpha Vantage has strict rate limits; if you hit limits, add delays/retries in the extraction notebooks or run them in chunks.

## 📁 Project Structure

```
tau-final-project/
├── Extract_SnP_500_index_data.ipynb        # S&P 500 price data extraction (SPY)
├── Extract_News_Sentiment_data.ipynb       # Alpha Vantage news sentiment extraction
├── pre_processing_1_min.ipynb              # Preprocess 1-min data
├── pre_processing_5_min.ipynb              # Preprocess 5-min data
├── pre_processing_15_min.ipynb             # Preprocess 15-min data
├── pre_processing_media_data.ipynb         # Aggregate media/news features
├── Feature_Selection_1min_data.ipynb       # PCA/feature selection (1-min)
├── Feature_Selection_5min_data.ipynb       # PCA/feature selection (5-min)
├── Feature_Selection_15min_data.ipynb      # PCA/feature selection (15-min)
├── LSTM_1_min.ipynb                        # LSTM training/eval (1-min)
├── LSTM_5_min.ipynb                        # LSTM training/eval (5-min)
├── LSTM_15_min.ipynb                       # LSTM training/eval (15-min)
├── Final_Report.ipynb                      # Consolidated report and results
├── requirements.txt                        # Python dependencies
└── data/                                   # Large data (not tracked in Git)
    ├── alpha_vantage/                      # Raw SPY and company news data
    ├── DWH/                                # Processed data warehouse
    ├── STG/                                # Staging data
    └── trained_models/                     # Saved model artifacts by timeframe

```

### Data layout examples (paths under `data/`)
```
alpha_vantage/
├── SPY/
│   ├── interval=1min/
│   ├── interval=5min/
│   └── interval=15min/
└── news_data/
    ├── AAPL/ ...
    ├── MSFT/ ...
    └── NVDA/ ...    # 40+ tickers

DWH/
├── features_1min_csv/
├── new_features_1min_csv/
├── new_features_5min_csv/
└── new_features_15min_csv/

trained_models/
├── 1_min/
├── 5_min/
└── 15_min/   # multiple timestamped runs
```

## 🔄 Usage Workflow

### 1. Data Extraction
```bash
# Extract S&P 500 price data
jupyter notebook Extract_SnP_500_index_data.ipynb

# Extract news sentiment data
jupyter notebook Extract_News_Sentiment_data.ipynb
```

### 2. Data Processing
```bash
# Process data for different time intervals
jupyter notebook pre_processing_1_min.ipynb
jupyter notebook pre_processing_5_min.ipynb
jupyter notebook pre_processing_15_min.ipynb
jupyter notebook pre_processing_media_data.ipynb
```

### 3. Feature Engineering
```bash
# Select optimal features using PCA
jupyter notebook Feature_Selection_1min_data.ipynb
jupyter notebook Feature_Selection_5min_data.ipynb
jupyter notebook Feature_Selection_15min_data.ipynb
```

### 4. Model Training & Evaluation
```bash
# Train LSTM models
jupyter notebook LSTM_1_min.ipynb
jupyter notebook LSTM_5_min.ipynb
jupyter notebook LSTM_15_min.ipynb
```

### Using saved models
- Trained artifacts are stored under `data/trained_models/<timeframe>/<run_timestamp>/`.
- Notebooks load/save models to these folders; adjust paths in the first config cell if needed.

## 📈 Trading Strategy

The system implements a simple yet effective trading strategy:

- **BUY Signal**: Predicted price change ≥ +0.5%
- **SELL Signal**: Predicted price change ≤ -0.5%
- **HOLD**: Price change between -0.5% and +0.5%

## 🔬 Technical Details

### Data Sources
- **Financial Data**: Alpha Vantage API (SPY index)
- **News Data**: Alpha Vantage News API (44 S&P 500 companies)
- **Time Range**: 2022-2023 market data

### Feature Categories
1. **Technical Indicators**: OHLCV, rolling statistics, volatility measures
2. **Temporal Features**: Hour, minute, day of week, trading session phases
3. **Sentiment Features**: News count, sentiment scores, relevance metrics
4. **Market Microstructure**: Volume patterns, price gaps, session transitions

### Model Architecture
- **Input**: 20 timesteps × 80+ features
- **Hidden Layers**: 2 LSTM layers (50 units each)
- **Output**: Single price prediction
- **Optimization**: Adam optimizer with MSE loss
- **Regularization**: MinMax scaling, dropout (where applicable)

### Environment notes
- The project targets Python 3.9. On macOS with Apple Silicon, `tensorflow-macos==2.15.0` is included in `requirements.txt`.
- For GPU acceleration on Apple Silicon, you may optionally install `tensorflow-metal`.
- Apache Spark runs in local mode via `findspark`; no cluster setup is required for the notebooks.

## 📊 Key Dependencies

```
tensorflow==2.15.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
pyspark (via findspark==2.0.1)
```

## 📄 License

This project is part of academic research at Tel Aviv University. Please cite appropriately if using for research purposes.

## 🤝 Contributing

This is an academic project, but suggestions and improvements are welcome. Please feel free to open issues or submit pull requests.

---

*Disclaimer: This project is for educational and research purposes only. Not intended for actual trading or investment decisions.*
