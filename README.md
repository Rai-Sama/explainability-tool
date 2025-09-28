# Explainy 🧠✨
*A lightweight open-source explainability tool for AI/ML models.*

---

## 🚀 Overview
Explainy is a Python library that makes it simple to understand and interpret machine learning models.  
It provides **model-agnostic explanations** so you can debug, validate, and trust your AI.

---

## ✨ Features
- **Global Feature Importance** – Identify which features matter most overall.  
- **Local Explanations** – Understand why a single prediction was made.  
- **Counterfactuals** – Discover minimal changes needed to flip predictions.  
- **Interactive Visualization** – Explore results in a clean Streamlit UI.  

---

## 🔧 Usage Example
```python
from explainy import Explainer
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Create explainer
expl = Explainer(model, X_train, y_train)

# Global importance
print(expl.global_feature_importance())

# Local explanation
print(expl.local_explanation(X_test.iloc[0]))
```

---

## 🖥️ Streamlit Demo
Run the interactive demo with:
```bash
streamlit run examples/demo_streamlit.py
```

---

## 📂 Project Structure
```
explainy/
│── core/
│   ├── core.py           # Core methods of the package
│   ├── visualizer.py     # Plotting + helpers
│── examples/
│   ├── demo_streamlit.py # Interactive demo
│── README.md
```

---

## 🤝 Contributing
We welcome contributions! Feel free to open issues or submit PRs.  
Check out our [Contributing Guide](CONTRIBUTING.md) (coming soon).

---

## 📜 License
MIT License © 2025
