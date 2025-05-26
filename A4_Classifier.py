import os
from A4_datawrangling import *
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, CategoricalNB
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*Bins whose width are too small.*")

def transformer(classifier_name):
    if classifier_name == 'gaussian':
        return ColumnTransformer([
            ('num', StandardScaler(), numerical_cols)
            # categorical omitted
        ])
    elif classifier_name == 'multinomial':
        return ColumnTransformer([
            ('num', KBinsDiscretizer(n_bins=4), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    elif classifier_name == 'bernoulli':
        return ColumnTransformer([
            ('num', KBinsDiscretizer(n_bins=4), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    elif classifier_name == 'categorical':
        return ColumnTransformer([
            ('num', KBinsDiscretizer(n_bins=4), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    elif classifier_name in ['svm_linear', 'svm_rbf']:
        return ColumnTransformer([
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])
    elif classifier_name == 'random_forest':
        return ColumnTransformer([
            ('num', 'passthrough', numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ])

def evaluate_all_models(X, y, cv=5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    best_models = {}
    results = []

    models = {
        'gaussian': (GaussianNB(), {}),
        'multinomial': (MultinomialNB(), {'classifier__alpha': [0.1, 1.0, 10]}),
        'bernoulli': (BernoulliNB(), {'classifier__alpha': [0.1, 1.0], 'classifier__binarize': [0.0, 0.5]}),
        'categorical': (CategoricalNB(), {'classifier__alpha': [0.1, 1.0]}),
        'svm_linear': (SVC(kernel='linear', probability=True), {'classifier__C': [0.1, 1, 10]}),
        'svm_rbf': (SVC(kernel='rbf', probability=True), {'classifier__C': [0.1, 1, 10], 'classifier__gamma': ['scale', 0.01]}),
        'random_forest': (RandomForestClassifier(random_state=0), {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 5, 10]})
    }

    for name, (model, param_grid) in models.items():
        preprocessor = transformer(name)
    
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=0),
            scoring=['accuracy', 'recall_macro', 'precision_macro', 'f1_macro'],
            refit='accuracy',
            error_score='raise'
        )

        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_
        results.append({
            'Classifier': name,
            'Best Accuracy': grid.cv_results_['mean_test_accuracy'][grid.best_index_],
            'F1': grid.cv_results_['mean_test_f1_macro'][grid.best_index_],
            'Recall': grid.cv_results_['mean_test_recall_macro'][grid.best_index_],
            'Precision': grid.cv_results_['mean_test_precision_macro'][grid.best_index_],
            'Best Params': grid.best_params_
        })

    results_df = pd.DataFrame(results).sort_values(by='Best Accuracy', ascending=False)

    for name, model in best_models.items():
        y_pred = model.predict(X_test)

        print(f"\n📌 {name.upper()}")
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        visualize_classifier_pca(name, model, X_test, y_test, filepath=f"output/classifier/{name}.png")

    # visualize_predictions(best_models, X_test, y_test)
    # visualize_pca_predictions(best_models, X_test, y_test)

    return results_df, best_models

def visualize_classifier_pca(model_name, model, X, y, filepath):
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    X_transformed = model.named_steps['preprocessor'].transform(X)
    y_pred = model.predict(X)

    pca = PCA(n_components=2)
    y_pred_encoded = label_enc.transform(y_pred)
    X_pca = pca.fit_transform(X_transformed)

    plt.figure(figsize=(16, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pred_encoded, cmap=plt.cm.Set1)
    plt.title(f"{model_name.upper()}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    output_dir = os.path.dirname(filepath)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.savefig(filepath)
    plt.close()

if __name__ == "__main__":
    (results, models) = evaluate_all_models(feature_columns, target_column);
    print(results)
