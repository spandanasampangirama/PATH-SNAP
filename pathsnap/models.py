"""Model zoo: 6 models from 6 algorithm families.  Owner: P2

Every model is an sklearn estimator (wrapped in a Pipeline with its scaler where the algorithm is
scale-sensitive) exposing predict_proba -> P(candidate is the correct arc). These probabilities become
HMM emissions, so probability quality (log loss) matters, not only F1.

Class imbalance (~16% positives: one correct arc among ~6 candidates per point): we deliberately do NOT
re-weight classes by default. Re-weighting distorts predicted probabilities (hurts log loss and the HMM
emissions), and top-1 accuracy only depends on the ranking inside each point, which weighting barely
changes. Macro-F1 at threshold 0.5 is reported as required.

  name     family                     scaled   notes
  logreg   Linear / probabilistic     yes      baseline; calibrated-ish probabilities, fast
  gnb      Probabilistic (generative) no       assumes independent Gaussian features (they are not)
  knn      Instance-based             yes      distance-weighted; trained on a row subsample (memory/time)
  rf       Ensemble (bagging)         no       depth/leaf limits keep the saved model small
  lgbm     Boosting                   no       histogram gradient boosting
  mlp      Neural network             yes      early stopping; exposes loss_curve_ (epoch history)
"""
from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import CFG

MODEL_FAMILIES = {
    "logreg": "Linear / probabilistic",
    "gnb": "Probabilistic (generative)",
    "knn": "Instance-based",
    "rf": "Ensemble (bagging)",
    "lgbm": "Boosting",
    "mlp": "Neural network",
}

# Initial (untuned) hyperparameters — report section 3.2 B asks for these.
DEFAULT_PARAMS = {
    "logreg": {"C": 1.0, "max_iter": 2000},
    "gnb": {"var_smoothing": 1e-9},
    "knn": {"n_neighbors": 25, "weights": "distance"},
    "rf": {"n_estimators": 150, "max_depth": 18, "min_samples_leaf": 20, "max_features": "sqrt"},
    "lgbm": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 50,
             "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8},
    "mlp": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 1024,
            "max_iter": 60, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 6},
}

# Rows used to fit each model (absent = all). KNN prediction cost grows with training-set size.
# Value lives in configs/default.yaml -> training.max_train_rows (CONTRACTS §1: constants only in YAML).
MAX_TRAIN_ROWS = {k: int(v) for k, v in (CFG.get("training", {}).get("max_train_rows") or {}).items()}

OBJECTIVES = {
    "logreg": "binary cross-entropy (log loss) + L2 penalty",
    "gnb": "maximum likelihood of class-conditional Gaussians (no iterative loss)",
    "knn": "none (lazy learner; vote weighted by inverse distance)",
    "rf": "Gini impurity per split; bagged trees",
    "lgbm": "binary log loss, gradient boosting",
    "mlp": "binary cross-entropy (log loss) + L2, Adam optimiser",
}


def build_model(name: str, params: dict | None = None, seed: int = 42):
    """Return an unfitted sklearn estimator/Pipeline for `name`, with `params` overriding defaults."""
    p = {**DEFAULT_PARAMS[name], **(params or {})}
    if name == "logreg":
        return Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(random_state=seed, **p))])
    if name == "gnb":
        return GaussianNB(**p)
    if name == "knn":
        return Pipeline([("scale", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=-1, **p))])
    if name == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=seed, **p)
    if name == "lgbm":
        return LGBMClassifier(random_state=seed, n_jobs=-1, verbose=-1, **p)
    if name == "mlp":
        return Pipeline([("scale", StandardScaler()), ("clf", MLPClassifier(random_state=seed, **p))])
    raise ValueError(f"unknown model {name!r}; choose from {list(MODEL_FAMILIES)}")
