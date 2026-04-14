# Turkey Inflation Forecast Model

A machine learning project that forecasts Turkey's inflation trend using historical World Bank data and linear regression.

## Features

- Uses real inflation data for Turkey
- Trains a linear regression model
- Forecasts future inflation values
- Interactive Streamlit web app
- Adjustable forecast horizon

## Tech Stack

- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Data Source

World Bank inflation dataset for Turkey.

## Run Locally

```bash
git clone https://github.com/ipekpala/inflation-prediction-model.git
cd inflation-prediction-model
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
## Live App

https://inflation-prediction-model-6zn9q6rmcqplhcxardbauk.streamlit.app