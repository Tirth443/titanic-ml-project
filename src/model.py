from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier


def get_model(model_name="gradient_boosting"):

    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )

    elif model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            class_weight='balanced',
            random_state=42
        )

    else:
        raise ValueError("Invalid model name")