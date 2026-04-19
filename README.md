# Crop Yield Prediction App

A machine learning web application that predicts maize crop yield (bags per hectare) for farmers in Southwestern Nigeria. Built with Streamlit.

![Crop Yield Predictor](https://img.shields.io/badge/Streamlit-1.55.0-blue) ![Python](https://img.shields.io/badge/Python-3.14+-green) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange)

## Features

- Predicts maize yield based on farm conditions and management practices
- User-friendly interface for farmers
- Personalized recommendations to improve yield
- Confidence intervals for predictions

## Quick Start

### Prerequisites

- Python 3.14 or higher
- Git (to clone the repository)

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/kishimi-ctrl/crop-yield-prediction.git
   cd crop-yield-prediction
   ```

2. **Install uv** (if not already installed)

   ```bash
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex
   ```

3. **Install dependencies**

   ```bash
   uv sync
   ```

### Running the App

```bash
uv run streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

## Project Structure

```
crop-yield-prediction/
├── app/
│   ├── app.py           # Main Streamlit application
│   └── model_utils.py   # Prediction engine
├── models/
│   ├── rf_model_tuned.pkl     # Trained model
│   └── label_encoders.pkl     # Feature encoders
├── farm_data_v2.csv     # Training data
├── pyproject.toml       # Project dependencies
└── README.md           # This file
```

## Usage

1. Open the app in your browser
2. Fill in your farm details:
   - Farm size and experience
   - Soil type and water retention
   - Seed type and planting month
   - Irrigation and fertilizer usage
   - Pest management practices
   - Expected rainfall and flooding risk
3. Click "Predict My Yield"
4. View your predicted yield and recommendations


## Technologies Used

- **Python** - Programming language
- **Streamlit** - Web framework
- **scikit-learn** - Machine learning
- **pandas/numpy** - Data processing
- **matplotlib/seaborn** - Visualization
