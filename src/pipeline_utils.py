import lightgbm as lgb
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer


# Function to calculate the average rides over the last 4 weeks
def average_rides_last_4_weeks(X: pd.DataFrame) -> pd.DataFrame:
    last_4_weeks_columns = [
        f"rides_t-{7*24}",  # 1 week ago
        f"rides_t-{14*24}",  # 2 weeks ago
        f"rides_t-{21*24}",  # 3 weeks ago
        f"rides_t-{28*24}",  # 4 weeks ago
    ]

    # Ensure the required columns exist in the DataFrame
    for col in last_4_weeks_columns:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    # Calculate the average of the last 4 weeks
    X["average_rides_last_4_weeks"] = X[last_4_weeks_columns].mean(axis=1)

    return X


# FunctionTransformer to add the average rides feature
add_feature_average_rides_last_4_weeks = FunctionTransformer(
    average_rides_last_4_weeks, validate=False
)


# Custom transformer to add temporal features
class TemporalFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_["hour"] = X_["pickup_hour"].dt.hour
        X_["day_of_week"] = X_["pickup_hour"].dt.dayofweek

        return X_.drop(columns=["pickup_hour", "pickup_location_id"])


# Instantiate the temporal feature engineer
add_temporal_features = TemporalFeatureEngineer()


# Custom transformer to add FFT features
class FFTFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, time_series_col, new_feature_prefix, n_components=10):
        self.time_series_col = time_series_col
        self.new_feature_prefix = new_feature_prefix
        self.n_components = n_components

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        # 1. Perform FFT
        time_series = X_[self.time_series_col].values
        fft_result = fft(time_series)
        fft_magnitude = np.abs(fft_result)

        # 2. Find the dominant frequencies (excluding the DC component)
        indices = np.argsort(fft_magnitude)[::-1][1:]

        # 3. Create new features based on the magnitudes of the dominant frequencies
        for i, index in enumerate(indices[: self.n_components]):  # Top N frequencies
            X_[f"{self.new_feature_prefix}_fft_magnitude_{i+1}"] = fft_magnitude[index]

        return X_


# Instantiate the FFT feature engineer
add_fft_features = FFTFeatureEngineer(time_series_col="pickup_hour", new_feature_prefix="rides")


# Function to return the pipeline
def get_pipeline(**hyper_params):
    """
    Returns a pipeline with optional parameters for LGBMRegressor.

    Parameters:
    ----------
    **hyper_params : dict
        Optional parameters to pass to the LGBMRegressor.

    Returns:
    -------
    pipeline : sklearn.pipeline.Pipeline
        A pipeline with feature engineering and LGBMRegressor.
    """
    pipeline = make_pipeline(
        add_feature_average_rides_last_4_weeks,
        add_temporal_features,
        add_fft_features,  # Add FFT feature engineering to the pipeline
        lgb.LGBMRegressor(**hyper_params),  # Pass optional parameters here
    )
    return pipeline
