# Smart Home Appliance Energy Consumption Prediction

A machine learning regression project that predicts household appliance energy consumption for a 10-minute interval using indoor temperature and humidity measurements, lighting energy use, and outdoor weather conditions.

## Project Overview

This project uses the **UCI Appliances Energy Prediction** dataset and a **Gradient Boosting Regressor** to estimate appliance energy consumption (`Appliances`) in Wh per 10-minute interval.

The project follows a classical machine learning workflow:

**EDA → Train/Test Split → Feature Identification → Preprocessing → Baseline Model → Cross-Validation → Hyperparameter Tuning → Feature Importance → Final Model → Test Evaluation → Streamlit Deployment**

The final model achieved the following performance on the held-out test set:

| Metric | Result |
|---|---:|
| MAE | 36.53 Wh |
| RMSE | 71.66 Wh |
| R² | 0.4869 |

Compared with the baseline model, the final tuned model reduced MAE from **47.09 Wh to 36.53 Wh** and improved R² from **0.2698 to 0.4869**.

## Dataset

**Dataset:** UCI Appliances Energy Prediction

The dataset contains household sensor and weather measurements collected at 10-minute intervals.

The model uses 25 numerical input features after excluding:

- `date`
- `rv1`
- `rv2`

The target variable is:

- `Appliances` — appliance energy consumption in Wh per 10-minute interval

Dataset source:

https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

## Machine Learning Model

The project uses a **Gradient Boosting Regressor**.

The preprocessing and trained model are stored together in:

```text
smart_home_appliance_energy_model.pkl
```

This allows the Streamlit application to load the complete trained pipeline and make predictions using the same preprocessing used during model development.

## Project Structure

```text
Smart-Home-Appliance-Energy-Prediction/
│
├── Smart_Home_Appliance_Energy_Prediction_Final_GradientBoosting_GitHub.ipynb
├── app.py
├── smart_home_appliance_energy_model.pkl
├── requirements.txt
├── run_app.bat
├── .streamlit/
│   └── config.toml
└── README.md
```

## Streamlit Application

Streamlit Application Link: https://smarthomeapplianceenergyprediction-abgvavnqqovhshwemf83dg.streamlit.app/

The project includes an interactive Streamlit application that allows users to enter household sensor and weather measurements and receive an estimated appliance energy consumption value.

The application provides:

- A user-friendly energy prediction interface
- Human-readable sensor names instead of raw dataset codes
- Key model-driver information
- Indoor environmental inputs
- Outdoor weather inputs
- Lighting energy input
- Predicted appliance energy consumption in Wh
- Equivalent energy consumption in kWh
- Model performance information

The application uses the trained `.pkl` model artifact and does not retrain the model.

## How to Run Locally

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Smart-Home-Appliance-Energy-Prediction
```

### 2. Install the required packages

```bash
pip install -r requirements.txt
```

### 3. Make sure the model artifact is present

The following file must be in the same directory as `app.py`:

```text
smart_home_appliance_energy_model.pkl
```

### 4. Start the Streamlit application

On Windows, you can double-click:

```text
run_app.bat
```

Or run:

```bash
streamlit run app.py --server.port 8503
```

The application will be available at:

```text
http://localhost:8503
```

## Model Evaluation

The final Gradient Boosting model was evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

The model achieved:

```text
MAE  : 36.53 Wh
RMSE : 71.66 Wh
R²   : 0.4869
```

The prediction error analysis shows that the model performs better for typical appliance-consumption values and has larger errors for some high-consumption observations.

## Important Limitation

The UCI dataset represents measurements from a **single low-energy house over approximately 4.5 months**. Therefore, the model should be interpreted as a dataset-specific demonstration of appliance energy prediction rather than as a universally generalizable model for every household.

## References

1. UCI Machine Learning Repository — Appliances Energy Prediction  
   https://archive.ics.uci.edu/dataset/374/appliances%2Benergy%2Bprediction

2. Candanedo, L. M., Feldheim, V., & Deramaix, D. (2017).  
   *Data driven prediction models of energy use of appliances in a low-energy house.*

## Author

**Trivikram Kambhampati**
