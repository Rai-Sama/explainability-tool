# explainy/visual.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

def plot_global_importance(df_importance, top_n=20, figsize=(8,5)):
    df = df_importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df['feature'], df['importance'])
    ax.set_xlabel("Importance")
    ax.set_title("Global feature importance")
    plt.tight_layout()
    return fig

def plot_shap_waterfall(shap_values, feature_names, idx=0, show=True):
    # shap_values: shap.Explanation or array-like
    try:
        shap.plots.waterfall(shap_values[idx])
    except Exception:
        # fallback: simple bar plot
        vals = np.array(shap_values[0])
        df = pd.DataFrame({'feature': feature_names, 'shap': vals}).sort_values('shap', key=abs, ascending=False)
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df['feature'], df['shap'])
        ax.set_xticklabels(df['feature'], rotation=90)
        plt.tight_layout()
        return fig
