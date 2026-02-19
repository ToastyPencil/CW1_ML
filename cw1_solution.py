import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)


print("Loading data ")
try:
    df_train = pd.read_csv('CW1_train.csv')
    df_test = pd.read_csv('CW1_test.csv') # Ensure this file is in the directory
    print(f"Training data shape: {df_train.shape}")
    print(f"Test data shape: {df_test.shape}")
except FileNotFoundError:
    print("Error: 'CW1_train.csv' or 'CW1_test.csv' not found.")
    # creating a dummy test set for demonstration if file is missing
    df_test = df_train.drop('outcome', axis=1).iloc[:100].copy() 


print("\nPerforming EDA ")

# Target Variable Distribution
plt.figure(figsize=(10, 6))
sns.histplot(df_train['outcome'], kde=True, bins=30)
plt.title('Distribution of Outcome Variable')
plt.xlabel('Outcome')
plt.ylabel('Frequency')
plt.savefig('eda_outcome_distribution.png')
print(" - Saved 'eda_outcome_distribution.png'")

#Correlation Matrix
plt.figure(figsize=(12, 10))
# Select only numerical columns for correlation
numerical_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
corr_matrix = df_train[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Matrix')
plt.savefig('eda_correlation_matrix.png')
print(" - Saved 'eda_correlation_matrix.png'")


print("\nSetting up preprocessing pipeline")

# Identify categorical and numerical columns
categorical_cols = ['cut', 'color', 'clarity']
numerical_cols = [c for c in df_train.columns if c not in categorical_cols + ['outcome']]

# Create transformers
# - OneHotEncoder for categorical variables (drop='first' to avoid multicollinearity)
# - StandardScaler for numerical variables (helps linear models converge faster)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ])


print("\nDefining models to test ")

# Dictionary of models to evaluate
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting (sklearn)': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, n_jobs=-1)
}

# Evaluate models using Cross-Validation
results = {}
print(f"{'Model':<30} | {'Mean R2':<10} | {'Std R2':<10}")
print("-" * 55)

X = df_train.drop('outcome', axis=1)
y = df_train['outcome']

for name, model in models.items():
    # Create a pipeline with preprocessing and the model
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    # Perform 5-fold Cross-Validation
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
    
    results[name] = cv_scores
    print(f"{name:<30} | {cv_scores.mean():.4f}     | {cv_scores.std():.4f}")

# Select the best model based on Mean R2
best_model_name = max(results, key=lambda k: results[k].mean())
print(f"\nBest performing model: {best_model_name}")


print(f"\nTuning hyperparameters for {best_model_name} ")

# 1. XGBoost
if best_model_name == 'XGBoost':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.7, 0.8, 1.0],
        'model__colsample_bytree': [0.7, 0.8, 1.0]
    }
    base_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)

# 2. Random Forest
elif best_model_name == 'Random Forest':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_split': [2, 5, 10]
    }
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)

# 3. Gradient Boosting (The missing block that caused the error)
elif best_model_name == 'Gradient Boosting (sklearn)':
    param_grid = {
        'model__n_estimators': [100, 200, 300],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7],
        'model__subsample': [0.7, 0.8, 1.0]
    }
    base_model = GradientBoostingRegressor(random_state=42)

# 4. Linear Models (Ridge, Lasso, Linear Regression)
else:
    if 'Ridge' in best_model_name:
        base_model = Ridge()
        param_grid = {'model__alpha': [0.1, 1.0, 10.0, 100.0]}
    elif 'Lasso' in best_model_name:
        base_model = Lasso()
        param_grid = {'model__alpha': [0.001, 0.01, 0.1, 1.0]}
    else:
        # Standard Linear Regression (no alpha)
        base_model = LinearRegression()
        param_grid = {'model__fit_intercept': [True, False]}

# Create pipeline for tuning
tuning_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', base_model)])

# Randomized Search
search = RandomizedSearchCV(tuning_pipeline, param_distributions=param_grid, 
                            n_iter=20, cv=5, scoring='r2', n_jobs=-1, random_state=42, verbose=1)

search.fit(X, y)

print(f"Best Parameters: {search.best_params_}")
print(f"Best CV R2 Score: {search.best_score_:.4f}")


print("\nTraining final model on full dataset ")
final_model = search.best_estimator_
final_model.fit(X, y)

print("Generating predictions on test set ")
# Generate predictions
# Note: Ensure df_test has the same columns as X (excluding outcome)
if 'outcome' in df_test.columns:
    df_test = df_test.drop('outcome', axis=1)
    
y_pred = final_model.predict(df_test)

# Create submission DataFrame
submission = pd.DataFrame({'yhat': y_pred})
submission.to_csv('CW1_submission_K24032130.csv', index=False)
print("Submission saved to 'CW1_submission_K24032130.csv'")



print("\nGenerating model evaluation plots for the report ")


try:
    # Get feature names from the one-hot encoder
    ohe_feature_names = final_model.named_steps['preprocessor']\
                                   .named_transformers_['cat']\
                                   .get_feature_names_out(categorical_cols)
    all_feature_names = numerical_cols + list(ohe_feature_names)
    
    # Get importances (works for XGBoost, Random Forest, Gradient Boosting)
    if hasattr(final_model.named_steps['model'], 'feature_importances_'):
        importances = final_model.named_steps['model'].feature_importances_
        
        # Create a DataFrame for plotting
        feat_imp = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances})
        feat_imp = feat_imp.sort_values(by='Importance', ascending=False).head(20) # Top 20
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=feat_imp, palette='viridis')
        plt.title(f'Top 20 Feature Importances ({best_model_name})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig('report_feature_importance.png')
        print(" - Saved 'report_feature_importance.png'")
except Exception as e:
    print(f"Could not generate feature importance plot: {e}")


y_train_pred = final_model.predict(X)

plt.figure(figsize=(8, 8))
plt.scatter(y, y_train_pred, alpha=0.3, color='blue')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) 
plt.xlabel('Actual Outcome')
plt.ylabel('Predicted Outcome')
plt.title(f'Predicted vs Actual ({best_model_name})')
plt.tight_layout()
plt.savefig('report_predicted_vs_actual.png')
print(" - Saved 'report_predicted_vs_actual.png'")


residuals = y - y_train_pred

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_train_pred, y=residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title(f'Residual Plot ({best_model_name})')
plt.tight_layout()
plt.savefig('report_residuals.png')
print(" - Saved 'report_residuals.png'")


print("\nAll report visualizations generated.")
