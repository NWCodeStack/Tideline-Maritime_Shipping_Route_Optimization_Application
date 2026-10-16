"""
ml_models.py - the Fuel Consumption Model.

Pipeline
--------
1. ``generate_voyage_dataset``  builds 5,000 synthetic voyage records
   (vessel_type, speed, draft, cargo_weight, wind_speed, wave_height, actual_fuel_burned).
2. ``train_fuel_model``         fits a scikit-learn RandomForestRegressor on it (a LinearRegression
   baseline is trained alongside, to show how much non-linearity the forest recovers).
3. ``save_model`` / ``load_model`` persist the fitted pipeline with ``pickle``.
4. ``FuelModel.predict_tonnes_per_day`` is what the routing engine calls, one batch per lane table.

Units
-----
speed_kn            knots through the water
draft_m             metres
cargo_weight_t      tonnes
wind_speed_kn       knots
wave_height_m       significant wave height, metres
actual_fuel_burned  tonnes of fuel per 24 h at sea (noon-report basis)

The "ground truth" used to synthesise the data follows ship-resistance physics: propulsion power
(and so fuel) grows with the CUBE of speed, with wetted surface (displacement^(2/3)), and with
added resistance from wind and waves, on top of a constant hotel load. The random forest is never
told this formula - it has to learn the curve from noisy observations.
"""
from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = Path(__file__).parent / "models" / "fuel_model.pkl"
FEATURES = ["vessel_type", "speed_kn", "draft_m", "cargo_weight_t", "wind_speed_kn", "wave_height_m"]
NUMERIC = FEATURES[1:]
TARGET = "actual_fuel_burned"
VESSEL_TYPES = ("Container", "Tanker")

# Hull and engine constants per vessel class (used only to synthesise training data).
HULL = {
    "Container": dict(k=0.01604, hotel_t=6.0, lightship_t=45_000, max_cargo_t=110_000, speed=(10.0, 25.0),
                      draft=(8.0, 14.5), cargo=(15_000, 110_000)),       # ~10,000 TEU
    "Tanker": dict(k=0.0185, hotel_t=4.0, lightship_t=30_000, max_cargo_t=160_000, speed=(8.0, 16.5),
                   draft=(9.0, 17.5), cargo=(40_000, 160_000)),           # Suezmax crude carrier
}


def true_fuel_tonnes_per_day(vessel_type, speed, cargo, wind, wave) -> np.ndarray:
    """Noise-free physics used to synthesise the dataset (never shown to the model)."""
    out = np.empty(len(speed))
    for vt, h in HULL.items():
        m = np.asarray(vessel_type) == vt
        if not m.any():
            continue
        displacement = (h["lightship_t"] + cargo[m]) / (h["lightship_t"] + h["max_cargo_t"])
        weather = 1.0 + 0.012 * wave[m] ** 2 + 0.0002 * wind[m] ** 2
        out[m] = h["hotel_t"] + h["k"] * speed[m] ** 3 * displacement ** (2.0 / 3.0) * weather
    return out


def generate_voyage_dataset(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Create ``n`` synthetic voyage records (half container ships, half tankers)."""
    rng = np.random.default_rng(seed)
    vt = rng.choice(VESSEL_TYPES, size=n)
    speed = np.empty(n)
    cargo = np.empty(n)
    draft = np.empty(n)
    for name, h in HULL.items():
        m = vt == name
        k = int(m.sum())
        speed[m] = rng.uniform(*h["speed"], k)
        cargo[m] = rng.uniform(*h["cargo"], k)
        load = (cargo[m] - h["cargo"][0]) / (h["cargo"][1] - h["cargo"][0])
        draft[m] = np.clip(h["draft"][0] + (h["draft"][1] - h["draft"][0]) * load + rng.normal(0, 0.25, k),
                           h["draft"][0], h["draft"][1] + 0.5)
    wind = np.clip(rng.gamma(shape=4.0, scale=4.0, size=n), 0.0, 60.0)                 # knots, mean ~16
    wave = np.clip(0.045 * wind ** 1.25 * rng.lognormal(0.0, 0.2, n), 0.2, 9.0)         # correlated with wind
    burn = true_fuel_tonnes_per_day(vt, speed, cargo, wind, wave) * rng.lognormal(0.0, 0.05, n)
    return pd.DataFrame({
        "vessel_type": vt, "speed_kn": speed.round(2), "draft_m": draft.round(2),
        "cargo_weight_t": cargo.round(0), "wind_speed_kn": wind.round(1), "wave_height_m": wave.round(2),
        TARGET: burn.round(2),
    })


def _make_pipeline(estimator) -> Pipeline:
    pre = ColumnTransformer(
        [("vessel", OneHotEncoder(categories=[list(VESSEL_TYPES)], handle_unknown="ignore"), ["vessel_type"])],
        remainder="passthrough",
    )
    return Pipeline([("prep", pre), ("model", estimator)])


@dataclass
class FuelModel:
    """A fitted fuel model plus the metrics and metadata we want to keep next to it."""
    pipeline: Pipeline
    metrics: dict = field(default_factory=dict)
    sklearn_version: str = sklearn.__version__

    def predict_tonnes_per_day(self, vessel_type, speed_kn, draft_m, cargo_weight_t, wind_speed_kn,
                               wave_height_m) -> np.ndarray:
        """Vectorised prediction. Scalars broadcast; vessel_type may be a single string."""
        arrays = np.broadcast_arrays(speed_kn, draft_m, cargo_weight_t, wind_speed_kn, wave_height_m)
        n = arrays[0].size
        frame = pd.DataFrame({
            "vessel_type": np.broadcast_to(np.asarray(vessel_type), (n,)).ravel() if n else [],
            "speed_kn": arrays[0].ravel().astype(float), "draft_m": arrays[1].ravel().astype(float),
            "cargo_weight_t": arrays[2].ravel().astype(float), "wind_speed_kn": arrays[3].ravel().astype(float),
            "wave_height_m": arrays[4].ravel().astype(float),
        })[FEATURES]
        return self.pipeline.predict(frame)

    def speed_curve(self, vessel_type: str, speeds, draft_m: float, cargo_weight_t: float,
                    wind_speed_kn: float = 12.0, wave_height_m: float = 1.2) -> np.ndarray:
        """Predicted fuel burn across a range of speeds in fixed conditions."""
        speeds = np.asarray(speeds, dtype=float)
        return self.predict_tonnes_per_day(vessel_type, speeds, draft_m, cargo_weight_t, wind_speed_kn, wave_height_m)


def train_fuel_model(df: pd.DataFrame | None = None, seed: int = 42, n_estimators: int = 150) -> FuelModel:
    """Fit the RandomForest fuel model. Takes ~1-2 seconds on a laptop."""
    if df is None:
        df = generate_voyage_dataset(seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=seed)

    forest = _make_pipeline(RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1,
                                                  random_state=seed))
    linear = _make_pipeline(LinearRegression())
    forest.fit(X_train, y_train)
    linear.fit(X_train, y_train)

    pred = forest.predict(X_test)
    metrics = {
        "n_samples": int(len(df)),
        "rf_r2": float(r2_score(y_test, pred)),
        "rf_mae_t_per_day": float(mean_absolute_error(y_test, pred)),
        "rf_mape_pct": float(np.mean(np.abs(pred - y_test) / y_test) * 100),
        "linear_r2": float(r2_score(y_test, linear.predict(X_test))),
    }
    # Refit on everything so the saved model sees all 5,000 voyages.
    forest.fit(df[FEATURES], df[TARGET])
    return FuelModel(pipeline=forest, metrics=metrics)


def save_model(model: FuelModel, path: Path = MODEL_PATH) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_model(path: Path = MODEL_PATH) -> FuelModel | None:
    """Load a pickled model; returns None if missing, unreadable, or from a different scikit-learn."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            model = pickle.load(fh)
    except Exception:                                     # corrupt file or incompatible pickle
        return None
    if not isinstance(model, FuelModel) or model.sklearn_version != sklearn.__version__:
        return None
    return model


def load_or_train_model(path: Path = MODEL_PATH) -> FuelModel:
    """What the app calls: reuse the pickle if it is valid, otherwise train and save a fresh one."""
    model = load_model(path)
    if model is None:
        model = train_fuel_model()
        try:
            save_model(model, path)
        except OSError:                                   # read-only checkout: keep going in memory
            pass
    try:                                                  # small per-lane batches: avoid joblib worker threads
        model.pipeline.set_params(model__n_jobs=1)
    except (ValueError, AttributeError):
        pass
    return model


def main() -> None:
    mdl = train_fuel_model()
    out = save_model(mdl)
    m = mdl.metrics
    print(f"Trained on {m['n_samples']} voyages -> {out}")
    print(f"  RandomForest  R2={m['rf_r2']:.4f}  MAE={m['rf_mae_t_per_day']:.2f} t/day  MAPE={m['rf_mape_pct']:.1f}%")
    print(f"  Linear base   R2={m['linear_r2']:.4f}  (the gap is the non-linearity the forest learned)")
    for vt, d, c in (("Container", 13.0, 90_000), ("Tanker", 16.0, 130_000)):
        lo, hi = HULL[vt]["speed"]
        curve = mdl.speed_curve(vt, [lo + 2, (lo + hi) / 2, hi - 1], d, c)
        print(f"  {vt:<9} burn at {lo + 2:.0f} / {(lo + hi) / 2:.1f} / {hi - 1:.0f} kn: "
              + " / ".join(f"{x:.0f}" for x in curve) + " t/day")


if __name__ == "__main__":
    # Re-import under the real module name so the pickle references ml_models.FuelModel, not __main__.FuelModel.
    from ml_models import main as _main
    _main()
