"""sklearn Pipeline and ColumnTransformer builder."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


NUMERICAL_FEATURES = ["credit_score", "annual_mileage"]
ORDINAL_FEATURES = ["age", "gender", "vehicle_ownership", "married", "children"]
COUNT_FEATURES = ["speeding_violations", "duis", "past_accidents"]
CATEGORICAL_FEATURES = ["driving_experience", "education", "income",
                        "vehicle_year", "vehicle_type"]

ALL_FEATURES = NUMERICAL_FEATURES + ORDINAL_FEATURES + COUNT_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor(model_type: str = "tree") -> ColumnTransformer:
    """Build a ColumnTransformer for the given model family.

    model_type='linear' : StandardScaler on numerics, OneHotEncoder for categoricals
    model_type='tree'   : No scaling, OrdinalEncoder for categoricals
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
    from sklearn.pipeline import Pipeline as SkPipeline

    if model_type not in ("tree", "linear"):
        raise ValueError(f"model_type must be 'tree' or 'linear', got '{model_type}'")

    if model_type == "linear":
        num_transformer = SkPipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        ord_transformer = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler",  StandardScaler()),
        ])
        count_transformer = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("scaler",  StandardScaler()),
        ])
        cat_transformer = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
    else:
        num_transformer   = SimpleImputer(strategy="median")
        ord_transformer   = SimpleImputer(strategy="most_frequent")
        count_transformer = SimpleImputer(strategy="most_frequent")
        cat_transformer   = SkPipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
        ])

    return ColumnTransformer(transformers=[
        ("num",   num_transformer,   NUMERICAL_FEATURES),
        ("ord",   ord_transformer,   ORDINAL_FEATURES),
        ("count", count_transformer, COUNT_FEATURES),
        ("cat",   cat_transformer,   CATEGORICAL_FEATURES),
    ], remainder="drop")


def build_pipeline(estimator, model_type: str = "tree") -> Pipeline:
    """Combine preprocessor and classifier into a single sklearn Pipeline."""
    preprocessor = build_preprocessor(model_type=model_type)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier",   estimator),
    ])
