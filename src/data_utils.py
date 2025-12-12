import pandas as pd
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)


def load_data(path: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded data: %s rows, %s cols", df.shape[0], df.shape[1])
    return df


def save_model(model, out_dir: str, name: str = "best_model_xgb.joblib"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def save_csv(df, out_dir: str, name: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    df.to_csv(path, index=False)
    logger.info("Wrote CSV: %s (shape=%s)", path, df.shape)
    return path
