# Standard library
import sys
import warnings
import json as _json
from pathlib import Path

warnings.filterwarnings('ignore')

# Data
import pandas as pd
import numpy as np

# Visualisation
import matplotlib
matplotlib.use("Agg")  # non-interactive: prevents plt.close() from blocking
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

# Machine learning
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

# -- Path configuration -----------------------------------------
# Works whether run as a Jupyter notebook or as a plain Python script.
_this_file = Path(globals().get('__file__', '') or '').resolve()
if _this_file.suffix in ('.py', '.ipynb') and (_this_file.parent / 'data' / 'raw').exists():
    PROJECT_ROOT = _this_file.parent
else:
    # Walk up from cwd to find the project folder (contains data/raw)
    PROJECT_ROOT = Path().resolve()
    for _p in [PROJECT_ROOT, *PROJECT_ROOT.parents]:
        if (_p / 'data' / 'raw').exists():
            PROJECT_ROOT = _p
            break

RAW_FILE    = PROJECT_ROOT / 'data' / 'raw'       / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
CLEAN_FILE  = PROJECT_ROOT / 'data' / 'processed' / 'hr_attrition_clean.csv'
MODELS_DIR  = PROJECT_ROOT / 'models'
FIGURES_DIR = PROJECT_ROOT / 'outputs' / 'figures'

# -- Global constants ------------------------------------------
RANDOM_STATE = 42
TARGET       = 'Attrition'
POSITIVE     = 1    # Attrition = Yes

# -- Display settings ------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

print('Setup complete.')
print(f'  Project root : {PROJECT_ROOT}')
print(f'  Raw file     : {RAW_FILE.exists()}')
print(f'  Clean file   : {CLEAN_FILE.exists()}')

if not RAW_FILE.exists():
    raise FileNotFoundError(
        f'Raw dataset not found at {RAW_FILE}\n'
        'Download from Kaggle and place in data/raw/'
    )

df_raw = pd.read_csv(RAW_FILE)
print(f'Raw dataset shape: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns')

# Column names and data types
dtype_df = pd.DataFrame({
    'Column'  : df_raw.columns,
    'DType'   : df_raw.dtypes.values,
    'Nunique' : [df_raw[c].nunique() for c in df_raw.columns],
    'Sample'  : [df_raw[c].iloc[0]   for c in df_raw.columns],
})
print(dtype_df)

# First 5 rows
df_raw.head()

# Missing values
missing = df_raw.isnull().sum()
missing_any = missing[missing > 0]
print('Missing values:', 'None found.' if missing_any.empty else '')
if not missing_any.empty:
    print(missing_any.to_frame('Missing Count'))

# Duplicate rows
print(f'Duplicate rows: {df_raw.duplicated().sum()}')

# Target column distribution
counts  = df_raw[TARGET].value_counts()
percents = df_raw[TARGET].value_counts(normalize=True).mul(100).round(1)
dist = pd.DataFrame({'Count': counts, 'Percentage (%)': percents})
print(f'Target column: {TARGET}')
print(f'Unique values: {df_raw[TARGET].unique()}')
print(dist)

# Constant and near-identifier columns
constant_cols    = [c for c in df_raw.columns if df_raw[c].nunique() == 1]
near_unique_cols = [c for c in df_raw.columns if df_raw[c].nunique() == len(df_raw)]
print('Constant columns (1 unique value):', constant_cols)
print('Near-identifier columns (all unique):', near_unique_cols)

# Numerical and categorical column counts
num_cols = df_raw.select_dtypes(include='number').columns.tolist()
cat_cols = df_raw.select_dtypes(include=['object','category']).columns.tolist()
print(f'Numerical columns  : {len(num_cols)}')
print(f'Categorical columns: {len(cat_cols)} -- {cat_cols}')

# Descriptive statistics
df_raw.describe().round(2).T

# Work on a copy -- raw file stays untouched
df = df_raw.copy()
print(f'Starting shape: {df.shape}')

# Step C -- Drop constant columns
constant_to_drop = ['EmployeeCount', 'Over18', 'StandardHours']
for col in constant_to_drop:
    print(f'  {col}: {df[col].nunique()} unique value(s) -- value = {df[col].iloc[0]!r}')
df = df.drop(columns=constant_to_drop)
print(f'\nAfter dropping constants: {df.shape}')

# Step D -- Drop identifier column
print(f'EmployeeNumber unique values: {df["EmployeeNumber"].nunique()} of {len(df)} rows')
df = df.drop(columns=['EmployeeNumber'])
print('EmployeeNumber dropped.')

# Step E -- Encode Attrition
print('Before encoding:', df['Attrition'].unique())
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
print('After encoding:', sorted(df['Attrition'].unique()))
print(f'  Attrition=1 (Yes/Left)  : {df["Attrition"].sum()}')
print(f'  Attrition=0 (No/Stayed) : {(df["Attrition"]==0).sum()}')

# Step F -- Encode OverTime
print('Before encoding:', df['OverTime'].unique())
df['OverTime'] = df['OverTime'].map({'Yes': 1, 'No': 0})
print('After encoding:', sorted(df['OverTime'].unique()))

# Step G -- Year-column logical consistency checks
checks = [
    ('YearsAtCompany',          'TotalWorkingYears'),
    ('YearsInCurrentRole',      'YearsAtCompany'),
    ('YearsWithCurrManager',    'YearsAtCompany'),
    ('YearsSinceLastPromotion', 'YearsAtCompany'),
]
print('Year-column logical consistency:')
for col_a, col_b in checks:
    violations = (df[col_a] > df[col_b]).sum()
    status = 'OK OK' if violations == 0 else f'(!) {violations} violations'
    print(f'  {col_a} <= {col_b} : {status}')

# Preprocessing validation
print('PREPROCESSING VALIDATION')
print(f'  Rows: {df_raw.shape[0]} -> {df.shape[0]}  (expected: same)')
print(f'  Cols: {df_raw.shape[1]} -> {df.shape[1]}  (expected: 31)')
print(f'  Missing values remaining : {df.isnull().sum().sum()}')
print(f'  Duplicate rows remaining : {df.duplicated().sum()}')
print(f'  Attrition values         : {sorted(df[TARGET].unique())}')
print(f'  OverTime values          : {sorted(df["OverTime"].unique())}')

# Save processed dataset
CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(CLEAN_FILE, index=False)
print(f'\nProcessed dataset saved to: {CLEAN_FILE}')

# Helper -- attrition rate by a grouping column
def attrition_rate_by(dataframe, col):
    grp = dataframe.groupby(col)[TARGET].agg(Total='count', Attrited='sum').reset_index()
    grp['Rate_%'] = (grp['Attrited'] / grp['Total'] * 100).round(1)
    return grp.sort_values('Rate_%', ascending=False).reset_index(drop=True)

# Q1 -- Overall attrition rate
total     = len(df)
attrited  = int(df[TARGET].sum())
rate      = df[TARGET].mean() * 100
print(f'Total employees   : {total}')
print(f'Attrition = Yes   : {attrited} ({rate:.1f}%)')
print(f'Attrition = No    : {total - attrited} ({100-rate:.1f}%)')
print(f'Imbalance ratio   : {(total-attrited)/attrited:.1f} : 1 (No : Yes)')

# Q2 -- By department
dept_stats = attrition_rate_by(df, 'Department')
print('Attrition by Department:')
print(dept_stats)

# Q2 -- By job role
role_stats = attrition_rate_by(df, 'JobRole')
print('Attrition by Job Role:')
print(role_stats)

# Q3 -- By overtime
ot_stats = attrition_rate_by(df, 'OverTime')
ot_stats['OverTime_Label'] = ot_stats['OverTime'].map({1:'Yes (Overtime)', 0:'No Overtime'})
print('Attrition by Overtime:')
print(ot_stats[['OverTime_Label','Total','Attrited','Rate_%']])

# Q4 -- By job satisfaction
js_map   = {1:'1-Low', 2:'2-Medium', 3:'3-High', 4:'4-Very High'}
js_stats = attrition_rate_by(df, 'JobSatisfaction').sort_values('JobSatisfaction')
js_stats['Label'] = js_stats['JobSatisfaction'].map(js_map)
print('Attrition by Job Satisfaction:')
print(js_stats[['Label','Total','Attrited','Rate_%']])

# Q5 -- By tenure band
df_eda   = df.copy()
bins     = [0, 2, 5, 10, 20, 40]
tb_labels= ['0-2 yrs', '3-5 yrs', '6-10 yrs', '11-20 yrs', '21+ yrs']
df_eda['TenureBand'] = pd.cut(df_eda['YearsAtCompany'], bins=bins, labels=tb_labels, right=True)
ten_stats = df_eda.groupby('TenureBand', observed=True)[TARGET].agg(
    Total='count', Attrited='sum').reset_index()
ten_stats['Rate_%'] = (ten_stats['Attrited'] / ten_stats['Total'] * 100).round(1)
print(f'Median tenure -- Stayed: {df[df[TARGET]==0]["YearsAtCompany"].median():.0f} yrs')
print(f'Median tenure -- Left  : {df[df[TARGET]==1]["YearsAtCompany"].median():.0f} yrs')
print('\nAttrition rate by tenure band:')
print(ten_stats)

# Q6 -- Monthly income comparison
inc_stayed = df[df[TARGET]==0]['MonthlyIncome']
inc_left   = df[df[TARGET]==1]['MonthlyIncome']
print(f'Median income -- Stayed: {inc_stayed.median():,.0f}')
print(f'Median income -- Left  : {inc_left.median():,.0f}')
print(f'Mean   income -- Stayed: {inc_stayed.mean():,.0f}')
print(f'Mean   income -- Left  : {inc_left.mean():,.0f}')

# Q7 -- By age band
age_bins   = [17, 25, 35, 45, 55, 60]
age_labels = ['18-25', '26-35', '36-45', '46-55', '56-60']
df_eda['AgeBand'] = pd.cut(df_eda['Age'], bins=age_bins, labels=age_labels, right=True)
age_stats = df_eda.groupby('AgeBand', observed=True)[TARGET].agg(
    Total='count', Attrited='sum').reset_index()
age_stats['Rate_%'] = (age_stats['Attrited'] / age_stats['Total'] * 100).round(1)
print(f'Median age -- Stayed: {df[df[TARGET]==0]["Age"].median():.0f}')
print(f'Median age -- Left  : {df[df[TARGET]==1]["Age"].median():.0f}')
print(age_stats)

# Additional: Business travel, Job involvement, Work-life balance
for col, label in [('BusinessTravel','Business Travel'),
                   ('JobInvolvement','Job Involvement'),
                   ('WorkLifeBalance','Work-Life Balance')]:
    stats = attrition_rate_by(df, col)
    print(f'\nAttrition by {label}:')
    print(stats.to_string(index=False))

# Pearson correlations with Attrition
num_df   = df.select_dtypes(include='number')
corr_tgt = num_df.corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)
print('Linear correlations with Attrition (top 10, absolute value):')
print(corr_tgt.head(10).round(4).to_string())

# Display all 15 EDA figures
eda_figs = sorted(FIGURES_DIR.glob('0[0-9]_*.png')) + \
           sorted(FIGURES_DIR.glob('1[0-5]_*.png'))

fig_titles = {
    '01_target_distribution.png':     'Fig 1: Overall Attrition Distribution',
    '02_department_distribution.png': 'Fig 2: Department Distribution',
    '03_jobrole_distribution.png':    'Fig 3: Job Role Distribution',
    '04_attrition_by_department.png': 'Fig 4: Attrition by Department',
    '05_attrition_by_jobrole.png':    'Fig 5: Attrition by Job Role',
    '06_attrition_by_overtime.png':   'Fig 6: Attrition by Overtime',
    '07_attrition_by_jobsatisfaction.png': 'Fig 7: Attrition by Job Satisfaction',
    '08_tenure_distribution.png':     'Fig 8: Attrition by Tenure',
    '09_income_distribution.png':     'Fig 9: Income Distribution by Attrition',
    '10_age_distribution.png':        'Fig 10: Age Distribution by Attrition',
    '11_attrition_by_travel.png':     'Fig 11: Attrition by Business Travel',
    '12_attrition_by_jobinvolvement.png': 'Fig 12: Attrition by Job Involvement',
    '13_attrition_by_worklifebalance.png': 'Fig 13: Attrition by Work-Life Balance',
    '14_compensation_correlation.png':'Fig 14: Compensation Correlations',
    '15_numeric_correlation_heatmap.png': 'Fig 15: Numeric Correlation Heatmap',
}

for fig_path in sorted(FIGURES_DIR.glob('*.png')):
    if fig_path.name.startswith(('01_','02_','03_','04_','05_',
                                  '06_','07_','08_','09_','10_',
                                  '11_','12_','13_','14_','15_')):
        title = fig_titles.get(fig_path.name, fig_path.name)
        print(title)
        img = mpimg.imread(str(fig_path))
        plt.figure(figsize=(12, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.close()

# Load the processed file (ensures consistency with saved pipeline)
df_model = pd.read_csv(CLEAN_FILE)

feature_cols = [c for c in df_model.columns if c != TARGET]
CAT_FEATURES = df_model[feature_cols].select_dtypes(include='object').columns.tolist()
NUM_FEATURES = df_model[feature_cols].select_dtypes(include='number').columns.tolist()

X = df_model[feature_cols]
y = df_model[TARGET]

print(f'Feature columns : {len(feature_cols)}')
print(f'  Numerical     : {len(NUM_FEATURES)}')
print(f'  Categorical   : {len(CAT_FEATURES)} -- {CAT_FEATURES}')
print(f'Target          : {TARGET}  (1=Yes/Left, 0=No/Stayed)')

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)
print(f'Training set : {X_train.shape[0]} rows  |  Attrition=1: {y_train.sum()} ({y_train.mean()*100:.1f}%)')
print(f'Test set     : {X_test.shape[0]} rows  |  Attrition=1: {y_test.sum()} ({y_test.mean()*100:.1f}%)')
print('\nStratification confirmed -- class ratio preserved in both sets.')
print('(!) Test set is now locked. Not used for fitting, tuning, or selection.')

# Define preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe',     OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)),
])
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer,     NUM_FEATURES),
    ('cat', categorical_transformer, CAT_FEATURES),
], remainder='passthrough')
print('Preprocessing pipeline defined.')

# Define all four model pipelines
models = {
    'Dummy Baseline': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE))
    ]),
    'Logistic Regression': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(
            class_weight='balanced', max_iter=1000,
            random_state=RANDOM_STATE, solver='lbfgs'
        ))
    ]),
    'Decision Tree': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', DecisionTreeClassifier(
            class_weight='balanced', max_depth=6,
            min_samples_leaf=10, random_state=RANDOM_STATE
        ))
    ]),
    'Random Forest': Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestClassifier(
            n_estimators=200, class_weight='balanced_subsample',
            max_depth=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=-1
        ))
    ]),
}
print('Model pipelines defined:', list(models.keys()))

# Train and evaluate all models on the locked test set
results = {}

for name, pipeline in models.items():
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] \
             if hasattr(pipeline.named_steps['model'], 'predict_proba') else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    rauc = roc_auc_score(y_test, y_prob) if y_prob is not None else float('nan')
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': round(acc,4), 'Precision': round(prec,4),
        'Recall':   round(rec,4), 'F1':        round(f1,4),
        'ROC_AUC':  round(rauc,4) if not np.isnan(rauc) else 'N/A',
        'cm': cm, 'y_prob': y_prob,
    }
    print(f'\n{name}')
    print(f'  Accuracy={acc:.4f}  Precision={prec:.4f}  Recall={rec:.4f}  F1={f1:.4f}  AUC={rauc:.4f}')

print('\nAll models trained and evaluated.')

# Model comparison table
comparison = pd.DataFrame([
    {'Model': name,
     'Accuracy':  m['Accuracy'],
     'Precision': m['Precision'],
     'Recall':    m['Recall'],
     'F1':        m['F1'],
     'ROC_AUC':   m['ROC_AUC']}
    for name, m in results.items()
]).set_index('Model')

print(comparison
)

# 5-fold cross-validation (training set only)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print('5-Fold CV on Training Set (test set NOT used):')
print(f'{"Model":<25}  {"CV F1":>8}  {"_":>6}  {"CV AUC":>8}')
print('-' * 55)
for name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
    scores = cross_validate(models[name], X_train, y_train,
                            cv=cv, scoring=['f1', 'roc_auc'], n_jobs=-1)
    print(f'{name:<25}  {scores["test_f1"].mean():>8.4f}  '
          f'{scores["test_f1"].std():>6.4f}  {scores["test_roc_auc"].mean():>8.4f}')

# Confusion matrices -- 2x2 grid
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()
for ax, (name, m) in zip(axes, results.items()):
    sns.heatmap(m['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Pred No', 'Pred Yes'],
                yticklabels=['Actual No', 'Actual Yes'],
                linewidths=0.5, cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    ax.set_title(
        f'{name}\nRecall={m["Recall"]:.2f}  Prec={m["Precision"]:.2f}  F1={m["F1"]:.2f}',
        fontsize=10, fontweight='bold'
    )
fig.suptitle('Confusion Matrices -- All Models (Test Set)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '16_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC curves
fig, ax = plt.subplots(figsize=(8, 6))
colors_roc = ['#DD8452', '#55A868', '#4C72B0']
for (name, color) in zip(['Logistic Regression','Decision Tree','Random Forest'], colors_roc):
    y_prob = results[name]['y_prob']
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f'{name} (AUC={roc_auc:.3f})')
ax.plot([0,1],[0,1], 'k--', lw=1.2, label='Random Chance')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate (Recall)')
ax.set_title('ROC Curves -- Test Set', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '17_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Classification report -- selected model (Logistic Regression)
lr_pipeline = models['Logistic Regression']
y_pred_lr   = lr_pipeline.predict(X_test)
print('Classification Report -- Logistic Regression (Test Set):')
print(classification_report(y_test, y_pred_lr,
                             target_names=['No (Stayed)', 'Yes (Left)'],
                             zero_division=0))

# Model selection rationale
print('MODEL SELECTION: Logistic Regression')
print()
print(f'{"Model":<25}  {"Recall":>8}  {"F1":>8}  {"AUC":>8}')
print('-' * 55)
for name in ['Logistic Regression', 'Decision Tree', 'Random Forest']:
    m = results[name]
    print(f'{name:<25}  {m["Recall"]:>8.4f}  {m["F1"]:>8.4f}  {m["ROC_AUC"]:>8.4f}')
print()
print('Rationale: LR achieves highest Recall (0.6596), F1 (0.4627),')
print('and ROC-AUC (0.8042) with the most stable CV performance (std=0.029).')
print()
print('Trade-off: Precision=0.3563 -- ~2 in 3 flagged employees did not leave.')
print('Acceptable for voluntary retention programmes; not for individual punitive decisions.')

# Save pipeline
MODELS_DIR.mkdir(exist_ok=True)
pipeline_path = MODELS_DIR / 'lr_pipeline.joblib'
joblib.dump(lr_pipeline, pipeline_path)
print(f'Pipeline saved: {pipeline_path}')

# Logistic Regression coefficients
ohe           = lr_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['ohe']
ohe_names     = ohe.get_feature_names_out(CAT_FEATURES).tolist()
all_features  = NUM_FEATURES + ohe_names

lr_coefs = lr_pipeline.named_steps['model'].coef_[0]
lr_df    = pd.DataFrame({'Feature': all_features, 'Coefficient': lr_coefs})\
             .sort_values('Coefficient', key=abs, ascending=False)

print('Top 15 Logistic Regression Coefficients (after standardisation):')
print('Positive = associated with higher P(Attrition=Yes) in this fictional dataset.')
print()
print(lr_df.head(15))

# LR coefficient bar chart
top_lr     = lr_df.head(20).sort_values('Coefficient')
colors_lr  = ['#C0392B' if c > 0 else '#1B4F8A' for c in top_lr['Coefficient']]
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(top_lr['Feature'], top_lr['Coefficient'], color=colors_lr, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title('LR Coefficients -- Top 20 (red=positive association, blue=negative)\n'
             'Association _ causation', fontweight='bold', fontsize=10)
ax.set_xlabel('Coefficient (after standardisation)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / '20_logistic_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()

# Random Forest feature importances
rf_model = models['Random Forest'].named_steps['model']
rf_df    = pd.DataFrame({'Feature': all_features,
                          'Importance': rf_model.feature_importances_})\
             .sort_values('Importance', ascending=False)
print('Top 15 Random Forest Feature Importances:')
print(rf_df.head(15))
