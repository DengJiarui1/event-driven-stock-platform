import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.config import DATA_PROCESSED, ARTIFACTS_MODELS


def main() -> None:
    ARTIFACTS_MODELS.mkdir(parents=True, exist_ok=True)

    X = np.load(DATA_PROCESSED / "X_lstm.npy")
    n_samples, seq_len, n_features = X.shape

    scaler = StandardScaler()
    X_reshaped = X.reshape(-1, n_features)
    X_scaled = scaler.fit_transform(X_reshaped).reshape(n_samples, seq_len, n_features)

    np.save(DATA_PROCESSED / "X_lstm_scaled.npy", X_scaled)
    joblib.dump(scaler, ARTIFACTS_MODELS / "lstm_scaler.pkl")

    print("LSTM 输入标准化完成")
    print(f"保存 scaler: {ARTIFACTS_MODELS / 'lstm_scaler.pkl'}")


if __name__ == "__main__":
    main()
