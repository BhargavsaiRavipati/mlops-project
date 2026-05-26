import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):
    # 🔥 IMPORTANT: adjust target column if needed
    target = df.columns[-1]

    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)