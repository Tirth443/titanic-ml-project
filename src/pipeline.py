from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import prepare_data
from model import get_model

import os
import pickle
import matplotlib.pyplot as plt


def run_pipeline():

    # Path
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'train.csv')

    # Load data
    X, y = prepare_data(data_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline (no scaler for tree models)
    pipeline = Pipeline([
        ('model', get_model())
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluation
    print("\n===== MODEL RESULTS =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Cross-validation
    scores = cross_val_score(pipeline, X, y, cv=5)
    print("\nCross-validation score:", scores.mean())

    # Save model FIRST - before plot to avoid crash
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    print("🔄 Saving model to:", model_path)
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print("✅ Model saved successfully!")

    # Feature importance plot (non-blocking & safe)
    try:
        model = pipeline.named_steps['model']
        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(10,6))
            plt.barh(X.columns, model.feature_importances_)
            plt.title("Feature Importance")
            plt.tight_layout()
            plot_path = os.path.join(os.path.dirname(__file__), '..', 'feature_importance.png')
            plt.savefig(plot_path)
            print(f"📊 Feature importance saved: {plot_path}")
        plt.close('all')  # Clean up
    except Exception as e:
        print(f"⚠️ Plot generation skipped (non-critical): {str(e)[:100]}")


if __name__ == "__main__":
    run_pipeline()
