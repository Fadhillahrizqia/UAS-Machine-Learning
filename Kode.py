"""
=============================================================================
KOMPARASI ALGORITMA BOOSTING UNTUK PREDIKSI DROPOUT MAHASISWA
dengan Interpretabilitas SHAP

Dataset : UCI Student Dropout and Academic Success
         (https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success)
         Kaggle mirror: https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention

Algoritma : XGBoost, LightGBM, CatBoost
Teknik    : SMOTE (imbalance handling), GridSearchCV (tuning), SHAP (interpretability)
=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import os

# ─── Konfigurasi ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
OUTPUT_DIR   = "/home/arfin/PycharmProjects/PythonProject/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("  PREDIKSI DROPOUT MAHASISWA: XGBoost vs LightGBM vs CatBoost")
print("=" * 65)

# ─── 1. LOAD DATASET ──────────────────────────────────────────────────────────
print("\n[1/7] Memuat dataset...")

# Download otomatis dari UCI via ucimlrepo jika tersedia, atau generate synthetic
try:
    import os
    local_path = os.path.join(os.path.dirname(__file__), 'data.csv')
    if os.path.exists(local_path):
        df = pd.read_csv(local_path, sep=';')  # UCI pakai separator titik koma
        X_raw = df.drop(columns=['Target'])
        y_raw = df['Target']
        print(f"    ✓ Dataset lokal dimuat: {X_raw.shape[0]} baris, {X_raw.shape[1]} fitur")
        source = "Lokal"
    else:
        from ucimlrepo import fetch_ucirepo
        dataset = fetch_ucirepo(id=697)
        X_raw = dataset.data.features
        y_raw = dataset.data.targets.squeeze()
        print(f"    ✓ Dataset UCI dimuat: {X_raw.shape[0]} baris, {X_raw.shape[1]} fitur")
        source = "UCI"
except Exception:
    # Fallback: buat data synthetic yang representatif
    print("    ⚠ UCI tidak tersedia, membuat data synthetic representatif...")
    np.random.seed(RANDOM_STATE)
    n = 4424

    dropout_mask   = np.random.choice([0,1,2], size=n, p=[0.32, 0.18, 0.50])
    # 0=Dropout, 1=Enrolled, 2=Graduate

    X_raw = pd.DataFrame({
        'Marital_status'              : np.random.choice([1,2,3,4,5,6], n),
        'Application_mode'            : np.random.choice(range(1,18), n),
        'Application_order'           : np.random.randint(0, 9, n),
        'Course'                      : np.random.choice(range(1,20), n),
        'Attendance'                  : np.random.choice([0,1], n),
        'Previous_qualification'      : np.random.choice(range(1,18), n),
        'Nationality'                 : np.random.choice(range(1,22), n),
        'Mothers_qualification'       : np.random.choice(range(1,7), n),
        'Fathers_qualification'       : np.random.choice(range(1,7), n),
        'Mothers_occupation'          : np.random.choice(range(0,10), n),
        'Fathers_occupation'          : np.random.choice(range(0,10), n),
        'Displaced'                   : np.random.choice([0,1], n),
        'Special_needs'               : np.random.choice([0,1], n, p=[0.97,0.03]),
        'Debtor'                      : np.random.choice([0,1], n, p=[0.88,0.12]),
        'Tuition_fees_uptodate'       : np.random.choice([0,1], n, p=[0.15,0.85]),
        'Gender'                      : np.random.choice([0,1], n),
        'Scholarship_holder'          : np.random.choice([0,1], n, p=[0.75,0.25]),
        'Age_at_enrollment'           : np.random.randint(17, 60, n),
        'International'               : np.random.choice([0,1], n, p=[0.97,0.03]),
        'Curricular_u1_credited'      : np.random.randint(0, 20, n),
        'Curricular_u1_enrolled'      : np.random.randint(0, 10, n),
        'Curricular_u1_evaluations'   : np.random.randint(0, 45, n),
        'Curricular_u1_approved'      : np.random.randint(0, 10, n),
        'Curricular_u1_grade'         : np.random.uniform(0, 18.9, n).round(2),
        'Curricular_u1_without_eval'  : np.random.randint(0, 12, n),
        'Curricular_u2_credited'      : np.random.randint(0, 20, n),
        'Curricular_u2_enrolled'      : np.random.randint(0, 10, n),
        'Curricular_u2_evaluations'   : np.random.randint(0, 45, n),
        'Curricular_u2_approved'      : np.random.randint(0, 10, n),
        'Curricular_u2_grade'         : np.random.uniform(0, 18.9, n).round(2),
        'Curricular_u2_without_eval'  : np.random.randint(0, 12, n),
        'Unemployment_rate'           : np.random.uniform(7, 17, n).round(1),
        'Inflation_rate'              : np.random.uniform(-0.8, 3.7, n).round(1),
        'GDP'                         : np.random.uniform(-4, 3.5, n).round(2),
    })
    labels = ['Dropout', 'Enrolled', 'Graduate']
    y_raw  = pd.Series([labels[i] for i in dropout_mask], name='Target')
    source = "Synthetic"
    print(f"    ✓ Data synthetic: {X_raw.shape[0]} baris, {X_raw.shape[1]} fitur")

# ─── 2. PREPROCESSING ─────────────────────────────────────────────────────────
print("\n[2/7] Preprocessing data...")

# Encode target: hanya fokus Dropout vs Non-Dropout (binary)
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw.astype(str))

# Buat binary: Dropout=1, lainnya=0
class_names = le.classes_
dropout_idx = list(class_names).index('Dropout') if 'Dropout' in class_names else 0
y_binary = (y_encoded == dropout_idx).astype(int)

print(f"    Distribusi kelas awal:")
unique, counts = np.unique(y_binary, return_counts=True)
for u, c in zip(unique, counts):
    label = 'Dropout' if u == 1 else 'Non-Dropout'
    print(f"      {label}: {c} ({c/len(y_binary)*100:.1f}%)")

# Handle missing values
X_clean = X_raw.fillna(X_raw.median(numeric_only=True))
X_clean = X_clean.select_dtypes(include=[np.number])
feature_names = X_clean.columns.tolist()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_binary, test_size=0.2, random_state=RANDOM_STATE, stratify=y_binary
)

# SMOTE untuk atasi imbalance
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print(f"\n    Setelah SMOTE:")
unique, counts = np.unique(y_train_sm, return_counts=True)
for u, c in zip(unique, counts):
    label = 'Dropout' if u == 1 else 'Non-Dropout'
    print(f"      {label}: {c}")

# Scaling (untuk konsistensi)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_sm)
X_test_sc  = scaler.transform(X_test)

print(f"\n    Train: {X_train_sc.shape[0]} | Test: {X_test_sc.shape[0]}")

# ─── 3. DEFINISI MODEL ────────────────────────────────────────────────────────
print("\n[3/7] Mendefinisikan model boosting...")

models = {
    'XGBoost': xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='logloss',
        random_state=RANDOM_STATE, verbosity=0
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbose=-1
    ),
    'CatBoost': cb.CatBoostClassifier(
        iterations=200, depth=5, learning_rate=0.1,
        random_seed=RANDOM_STATE, verbose=False
    )
}

# ─── 4. TRAINING & EVALUASI ───────────────────────────────────────────────────
print("\n[4/7] Training dan evaluasi model...")

results = {}
trained_models = {}
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    print(f"\n    ▶ {name}")
    model.fit(X_train_sc, y_train_sm)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    cv_scores = cross_val_score(model, X_train_sc, y_train_sm, cv=cv,
                                 scoring='accuracy', n_jobs=-1)

    metrics = {
        'Accuracy'       : accuracy_score(y_test, y_pred),
        'Precision'      : precision_score(y_test, y_pred, zero_division=0),
        'Recall'         : recall_score(y_test, y_pred, zero_division=0),
        'F1-Score'       : f1_score(y_test, y_pred, zero_division=0),
        'AUC-ROC'        : roc_auc_score(y_test, y_proba),
        'CV Accuracy'    : cv_scores.mean(),
        'CV Std'         : cv_scores.std(),
        'y_pred'         : y_pred,
        'y_proba'        : y_proba,
    }
    results[name]        = metrics
    trained_models[name] = model

    print(f"      Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"      Precision : {metrics['Precision']:.4f}")
    print(f"      Recall    : {metrics['Recall']:.4f}")
    print(f"      F1-Score  : {metrics['F1-Score']:.4f}")
    print(f"      AUC-ROC   : {metrics['AUC-ROC']:.4f}")
    print(f"      CV(10)    : {metrics['CV Accuracy']:.4f} ± {metrics['CV Std']:.4f}")

# ─── 5. VISUALISASI ───────────────────────────────────────────────────────────
print("\n[5/7] Membuat visualisasi...")

colors = {'XGBoost': '#2E86AB', 'LightGBM': '#A23B72', 'CatBoost': '#F18F01'}

# --- Fig 1: Tabel perbandingan metrik ---
fig, ax = plt.subplots(figsize=(12, 4))
ax.axis('off')
metric_keys = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Accuracy']
table_data  = [[name] + [f"{results[name][m]:.4f}" for m in metric_keys]
               for name in models]
col_labels  = ['Model'] + metric_keys
table = ax.table(cellText=table_data, colLabels=col_labels,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)
for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor('#2E4057')
        cell.set_text_props(color='white', fontweight='bold')
    elif c == 0:
        cell.set_facecolor('#E8F4FD')
    else:
        cell.set_facecolor('#FAFAFA')
plt.title('Tabel Perbandingan Metrik Evaluasi Model', fontsize=13,
          fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_comparison_table.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ fig1_comparison_table.png")

# --- Fig 2: Bar chart perbandingan ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
metric_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
x   = np.arange(len(metric_list))
w   = 0.25
ax  = axes[0]
for i, name in enumerate(models):
    vals = [results[name][m] for m in metric_list]
    ax.bar(x + i*w, vals, w, label=name, color=colors[name], alpha=0.85)
ax.set_xticks(x + w)
ax.set_xticklabels(metric_list, fontsize=9)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Score')
ax.set_title('Perbandingan Metrik Evaluasi', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# CV comparison
ax2 = axes[1]
cv_means = [results[n]['CV Accuracy'] for n in models]
cv_stds  = [results[n]['CV Std'] for n in models]
bars = ax2.bar(list(models.keys()), cv_means, color=list(colors.values()),
               yerr=cv_stds, capsize=5, alpha=0.85)
ax2.set_ylim(0.5, 1.05)
ax2.set_ylabel('CV Accuracy (10-Fold)')
ax2.set_title('Cross-Validation Accuracy ± Std', fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, cv_means):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.suptitle('Komparasi Performa: XGBoost vs LightGBM vs CatBoost', fontsize=13,
             fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_performance_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ fig2_performance_comparison.png")

# --- Fig 3: Confusion Matrix (3 model) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, name in zip(axes, models):
    cm = confusion_matrix(y_test, results[name]['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', ax=ax,
                cmap='Blues', xticklabels=['Non-DO','Dropout'],
                yticklabels=['Non-DO','Dropout'])
    ax.set_title(f'Confusion Matrix\n{name}', fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.suptitle('Confusion Matrix Ketiga Model', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ fig3_confusion_matrices.png")

# --- Fig 4: ROC Curve ---
fig, ax = plt.subplots(figsize=(8, 6))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, results[name]['y_proba'])
    auc = results[name]['AUC-ROC']
    ax.plot(fpr, tpr, color=colors[name], lw=2, label=f'{name} (AUC={auc:.4f})')
ax.plot([0,1],[0,1],'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Komparasi Tiga Model Boosting', fontweight='bold')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig4_roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ fig4_roc_curve.png")

# ─── 6. SHAP ANALYSIS ─────────────────────────────────────────────────────────
print("\n[6/7] Analisis SHAP (interpretabilitas)...")

# Gunakan model terbaik berdasarkan F1
best_name  = max(results, key=lambda n: results[n]['F1-Score'])
best_model = trained_models[best_name]
print(f"    Model terbaik (F1): {best_name}")

# Sampel untuk SHAP agar cepat
X_shap   = pd.DataFrame(X_test_sc, columns=feature_names).iloc[:300]

if best_name == 'CatBoost':
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
else:
    explainer   = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_shap)
    shap_vals   = shap_values if not isinstance(shap_values, list) else shap_values[1]

# SHAP Summary Plot (Bar)
fig, ax = plt.subplots(figsize=(10, 7))
mean_shap = np.abs(shap_vals).mean(axis=0)
top_idx   = np.argsort(mean_shap)[-15:]
top_feat  = [feature_names[i] for i in top_idx]
top_vals  = mean_shap[top_idx]
bars = ax.barh(top_feat, top_vals, color='#2E86AB', alpha=0.85)
ax.set_xlabel('Mean |SHAP Value|')
ax.set_title(f'Top 15 Fitur Paling Berpengaruh\n(SHAP - {best_name})', fontweight='bold')
ax.grid(axis='x', alpha=0.3)
for bar, val in zip(bars, top_vals):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig5_shap_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ fig5_shap_importance.png")

# SHAP Beeswarm
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_vals, X_shap, feature_names=feature_names,
                  max_display=15, show=False, plot_size=(10,7))
plt.title(f'SHAP Summary Plot - {best_name}', fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig6_shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.close()
print("    ✓ fig6_shap_beeswarm.png")

# ─── 7. SIMPAN HASIL ──────────────────────────────────────────────────────────
print("\n[7/7] Menyimpan hasil akhir...")

result_df = pd.DataFrame({
    'Metrik'  : ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'CV Accuracy (10-fold)'],
    'XGBoost' : [f"{results['XGBoost'][m]:.4f}"  for m in ['Accuracy','Precision','Recall','F1-Score','AUC-ROC','CV Accuracy']],
    'LightGBM': [f"{results['LightGBM'][m]:.4f}" for m in ['Accuracy','Precision','Recall','F1-Score','AUC-ROC','CV Accuracy']],
    'CatBoost': [f"{results['CatBoost'][m]:.4f}" for m in ['Accuracy','Precision','Recall','F1-Score','AUC-ROC','CV Accuracy']],
})
result_df.to_csv(f"{OUTPUT_DIR}/hasil_evaluasi.csv", index=False)
print("    ✓ hasil_evaluasi.csv")

# Simpan top SHAP features untuk paper
top15 = pd.DataFrame({'Fitur': top_feat, 'Mean_SHAP': top_vals}).sort_values('Mean_SHAP', ascending=False)
top15.to_csv(f"{OUTPUT_DIR}/shap_top_features.csv", index=False)
print("    ✓ shap_top_features.csv")

print("\n" + "=" * 65)
print("  RINGKASAN HASIL AKHIR")
print("=" * 65)
print(f"\n{'Metrik':<22} {'XGBoost':>10} {'LightGBM':>10} {'CatBoost':>10}")
print("-" * 55)
for m in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']:
    vals = [results[n][m] for n in models]
    print(f"  {m:<20} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f}")
print("-" * 55)
print(f"\n  Model Terbaik (F1-Score): {best_name}")
print(f"  Sumber data             : {source}")
print(f"\n  Output tersimpan di     : {OUTPUT_DIR}/")
print("=" * 65)
