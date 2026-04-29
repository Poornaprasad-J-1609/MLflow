import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# ✅ Disable model logging in autolog (important)
mlflow.sklearn.autolog(log_models=False)

# Load and prepare data
wine = load_wine()
X_train, X_test, y_train, y_test = train_test_split(
    wine.data, wine.target, test_size=0.2, random_state=42
)

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    # ✅ Convert to float to avoid schema warning
    input_example = pd.DataFrame(X_train[:5]).astype("float64")

    # ✅ Correct model logging (this creates model.pkl inside MLflow structure)
    mlflow.sklearn.log_model(
        model,
        name="model",
        input_example=input_example
    )

    # Optional: explicitly log metrics (cleaner)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    mlflow.log_metric("train_accuracy", train_score)
    mlflow.log_metric("test_accuracy", test_score)

    print(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")