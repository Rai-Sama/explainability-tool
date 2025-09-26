"""
Lightweight, model-agnostic explainability wrapper.
Supports SHAP for tree & kernel expl, LIME as fallback, and
simple feature importance / local explanations.
"""

from typing import Optional, Callable, Any
import numpy as np
import pandas as pd

# third-party
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.base import BaseEstimator

class Explainer:
    def __init__(self, model: BaseEstimator, X_train: pd.DataFrame, feature_names=None, predict_proba: bool = False):
        """
        model: fitted scikit-learn-like model (must have predict or predict_proba)
        X_train: DataFrame used for background distribution
        """
        self.model = model
        self.X_train = X_train.reset_index(drop=True)
        self.feature_names = feature_names or list(self.X_train.columns)
        self.predict_proba = predict_proba
        self._init_shap()

    def _model_predict(self, X: pd.DataFrame):
        if self.predict_proba and hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def _init_shap(self):
        # Try to choose a SHAP explainer: TreeExplainer for tree models, otherwise KernelExplainer
        try:
            # prefer TreeExplainer for speed if supported
            self.shap_explainer = shap.Explainer(self.model, self.X_train, feature_names=self.feature_names)
            self.shap_available = True
        except Exception:
            # fallback: KernelExplainer (slower)
            try:
                wrapped = lambda x: self._model_predict(pd.DataFrame(x, columns=self.feature_names))
                self.shap_explainer = shap.KernelExplainer(wrapped, shap.sample(self.X_train, min(50, len(self.X_train))))
                self.shap_available = True
            except Exception:
                self.shap_explainer = None
                self.shap_available = False

        # LIME fallback setup
        try:
            self.lime_explainer = LimeTabularExplainer(
                training_data=self.X_train.values,
                feature_names=self.feature_names,
                mode="classification" if self.predict_proba else "regression",
                discretize_continuous=True
            )
            self.lime_available = True
        except Exception:
            self.lime_explainer = None
            self.lime_available = False

    def global_feature_importance(self, method: str = "shap", nsamples: int = 100):
        if method == "shap" and self.shap_available:
            sample = self.X_train.sample(min(nsamples, len(self.X_train)))
            shap_values = self.shap_explainer(sample)
            # convert to 1D importance per feature
            shap_vals = shap_values.values if hasattr(shap_values, "values") else shap_values
            vals = np.abs(shap_vals).mean(axis=0)
            if vals.ndim > 1:
                vals = vals.mean(axis=1)
            return pd.DataFrame({"feature": self.feature_names, "importance": vals}).sort_values("importance", ascending=False)


    def explain_local(self, x: pd.Series, nsamples: int = 100):
        """
        Explain a single example.
        Returns a dict with 'shap' and/or 'lime' explanations where available.
        """
        out = {}
        X_df = pd.DataFrame([x.values], columns=self.feature_names)

        if self.shap_available:
            shap_vals = self.shap_explainer(X_df)
            # for readability return (feature, shap_value) pairs
            if hasattr(shap_vals, "values"):
                shap_arr = shap_vals.values[0]
            else:
                shap_arr = np.array(shap_vals[0])
            out['shap'] = dict(zip(self.feature_names, shap_arr.tolist()))

        if self.lime_available:
            predict_fn = (lambda data: self.model.predict_proba(data) if self.predict_proba else self.model.predict(data))
            exp = self.lime_explainer.explain_instance(x.values, predict_fn, num_features=min(len(self.feature_names), 10))
            out['lime'] = dict(exp.as_list())

        return out

    def summary(self, top_n: int = 10):
        return self.global_feature_importance().head(top_n)

    # Simple counterfactual-ish suggestion (very basic; for demo purposes)
    def simple_counterfactual(self, x: pd.Series, target_fn: Callable[[Any], bool], feature_constraints: Optional[dict] = None, max_changes: int = 3):
        """
        Greedy local search: change top features toward direction that flips target_fn.
        Not a production counterfactual method, but useful for quick demos.
        """
        feature_constraints = feature_constraints or {}
        baseline = x.copy()
        ranked = self.global_feature_importance()
        for feat in ranked.feature.tolist():
            if feat in feature_constraints and feature_constraints[feat] is None:
                continue
            # move feature by +/- 1 std in direction that improves target
            mean = self.X_train[feat].mean()
            std = self.X_train[feat].std() or 1.0
            for delta in [std, -std]:
                trial = baseline.copy()
                trial[feat] = trial[feat] + delta
                if target_fn(self._model_predict(pd.DataFrame([trial.values], columns=self.feature_names))):
                    return trial
        return None
