import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.feature_selection import (
    SelectKBest,
    f_classif
)

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

data = pd.read_csv(url)

data = data.drop(
    ["PassengerId", "Name", "Ticket", "Cabin"],
    axis=1
)

X = data.drop(
    "Survived",
    axis=1
)

y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

numerical_features = [
    "Age",
    "Fare",
    "SibSp",
    "Parch"
]

categorical_features = [
    "Sex",
    "Embarked",
    "Pclass"
]

numerical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "feature_selection",
            SelectKBest(
                score_func=f_classif
            )
        ),
        (
            "classifier",
            RandomForestClassifier(
                random_state=42
            )
        )
    ]
)

param_grid = [
    {
        "feature_selection__k": [5, 6],
        "classifier": [
            RandomForestClassifier(
                random_state=42
            )
        ],
        "classifier__n_estimators": [
            100,
            200
        ]
    },
    {
        "feature_selection__k": [5, 6],
        "classifier": [
            SVC(
                probability=True,
                random_state=42
            )
        ],
        "classifier__C": [
            1,
            10
        ]
    }
]

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,
    scoring="accuracy",
    verbose=2,
    n_jobs=1,
    error_score="raise"
)

print("Starting Grid Search...")

grid_search.fit(
    X_train,
    y_train
)

best_model = grid_search.best_estimator_

y_pred = best_model.predict(
    X_test
)

print(
    "\nBest Parameters:",
    grid_search.best_params_
)

print(
    "\nAccuracy on Test Set:",
    accuracy_score(
        y_test,
        y_pred
    )
)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        y_pred
    )
)
