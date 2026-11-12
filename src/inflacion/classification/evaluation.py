from __future__ import annotations

import pandas as pd
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             f1_score, precision_recall_fscore_support)

DIMENSIONS = ["MONETARIA", "SOCIODEMOGRAFICA", "INSTITUCIONAL", "ESTRUCTURAL"]


def summary(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """AMBIGUO cuenta como error en las métricas globales y se reporta aparte como cobertura."""
    decided = y_pred != "AMBIGUO"
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=DIMENSIONS, average="macro", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "cobertura": decided.mean(),
        "accuracy_decididos": accuracy_score(y_true[decided], y_pred[decided]) if decided.any() else float("nan"),
    }


def per_class(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=DIMENSIONS, zero_division=0)
    return pd.DataFrame({"precision": p, "recall": r, "f1": f, "soporte": s}, index=DIMENSIONS)


def confusion(y_true: pd.Series, y_pred: pd.Series) -> pd.DataFrame:
    labels = DIMENSIONS + ["AMBIGUO"]
    return pd.DataFrame(confusion_matrix(y_true, y_pred, labels=labels),
                        index=pd.Index(labels, name="humano"), columns=pd.Index(labels, name="llm"))
