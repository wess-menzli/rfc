import os
import sys
import json
import warnings
import logging
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# ---------------- Load files ----------------
def load_excel_list(file_path: str) -> pd.DataFrame:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(file_path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(file_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    df = df.fillna("")
    logger.info(f"Loaded {len(df)} rows from '{file_path}'")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df

# ---------------- Fixed Cross-Validation Function ----------------
def fixed_cross_val_score(model, X, y, scoring='f1'):
    """Fixed cross-validation that handles small datasets"""
    try:
        # Use fewer folds for small datasets
        n_splits = min(3, len(np.unique(y)))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return scores
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        return np.array([0])

# ---------------- Congés Pipeline (Improved) ----------------
class CongesPipeline:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_medians = {}
        self.label_encoder = LabelEncoder()
        
        # Optimized feature set
        self.categorical_features = ["Département", "Superieur", "job_title"]
        self.numeric_features = [
            "Totale_des_jours", "Solde", "Month_debut", "Weekday_debut",
            "Season", "Duration_category", "Balance_sufficient", "Weekend_start", 
            "High_season", "Days_ratio", "Is_long_leave", "Days_until_month_end",
            "Request_advance_days"
        ]

    def engineer_merge(self, df_dem: pd.DataFrame, df_sol: pd.DataFrame) -> pd.DataFrame:
        df_dem = df_dem.rename(columns={"Demandeur": "Employee"})
        if "Employee" not in df_sol.columns:
            df_sol = df_sol.rename(columns={"Nom_Prenom": "Employee"})
        df = df_dem.merge(df_sol, on="Employee", how="left")

        # More robust decision logic
        decision_cols = ["avis_manager", "Status_globale", "avis_PMO", "validation_RH"]
        for col in decision_cols:
            if col not in df.columns:
                df[col] = ""
        
        # Improved decision logic
        df["decision_binary"] = df[decision_cols].apply(
            lambda r: "approved" if any(
                "approuvé" in str(v).lower() or 
                "validé" in str(v).lower() or 
                "accepté" in str(v).lower() or 
                "oui" in str(v).lower() or 
                "ok" in str(v).lower() 
                for v in r if pd.notna(v)
            ) else "rejected",
            axis=1
        )
        
        # Use binary target
        df["y"] = self.label_encoder.fit_transform(df["decision_binary"].astype(str))
        dist = Counter(df["y"])
        logger.info(f"Distribution cible après engineering: {dist}")

        # Remove classes with <10 samples
        classes_to_remove = [cls for cls, count in dist.items() if count < 10]
        if classes_to_remove:
            logger.warning(f"Removing classes with insufficient samples: {classes_to_remove}")
            df = df[~df["y"].isin(classes_to_remove)]
            df["y"] = self.label_encoder.fit_transform(df["decision_binary"])
            logger.info(f"New distribution: {Counter(df['y'])}")

        if len(df["y"].unique()) < 2:
            logger.error("Cible mono-classe. Corrigez les libellés sources.")
            return df

        # Numeric features
        for col in ["Totale_des_jours", "Solde"]:
            df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0)

        # Date features
        for col in ["Date_debut_congé"]:
            df[col] = pd.to_datetime(df.get(col, pd.NaT), errors="coerce")

        # Extract request date if available, otherwise use current date as reference
        request_date_col = df.get("Created", None)
        if request_date_col is not None:
            try:
                request_date = pd.to_datetime(request_date_col, errors="coerce")
                # Use the minimum request date as reference if there are multiple
                reference_date = request_date.min() if not request_date.isna().all() else pd.Timestamp.now()
            except:
                reference_date = pd.Timestamp.now()
        else:
            reference_date = pd.Timestamp.now()

        df["Month_debut"] = df["Date_debut_congé"].dt.month.fillna(0).astype(int)
        df["Weekday_debut"] = df["Date_debut_congé"].dt.weekday.fillna(0).astype(int)
        df["Day_debut"] = df["Date_debut_congé"].dt.day.fillna(0).astype(int)
        
        # Enhanced date features
        df["Season"] = pd.cut(df["Month_debut"], bins=[0, 3, 6, 9, 12], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
        df["Duration_category"] = pd.cut(df["Totale_des_jours"], bins=[0, 3, 7, 14, 100], labels=[0, 1, 2, 3], include_lowest=True).astype(int)
        df["Balance_sufficient"] = (df["Solde"] >= df["Totale_des_jours"]).astype(int)
        df["Weekend_start"] = (df["Weekday_debut"] >= 5).astype(int)
        df["High_season"] = ((df["Month_debut"] >= 6) & (df["Month_debut"] <= 8)).astype(int)
        
        # New features for better signal
        df["Days_ratio"] = np.where(df["Solde"] > 0, df["Totale_des_jours"] / df["Solde"], 0).clip(0, 5)  # Tighter clip
        df["Is_long_leave"] = (df["Totale_des_jours"] > 7).astype(int)
        
        # Calculate days until month end
        df["Days_until_month_end"] = df["Date_debut_congé"].apply(
            lambda x: (pd.Timestamp(year=x.year, month=x.month, day=1) + pd.offsets.MonthEnd(1) - x).days 
            if pd.notna(x) else 0
        ).fillna(0)
        
        # Calculate request advance days
        df["Request_advance_days"] = (df["Date_debut_congé"] - reference_date).dt.days.fillna(0).clip(0, 365)

        # Handle categorical (reduce to top 5 for less overfitting)
        for c in self.categorical_features:
            if c not in df.columns:
                df[c] = "NA"
            df[c] = df[c].astype(str).fillna("NA")
            if df[c].nunique() > 5:
                top_values = df[c].value_counts().head(5).index
                df[c] = df[c].apply(lambda x: x if x in top_values else "Other")

        # Handle numeric
        for c in self.numeric_features:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce")
            self.feature_medians[c] = float(df[c].median()) if not df[c].dropna().empty else 0
            df[c] = df[c].fillna(self.feature_medians[c])

        before = len(df)
        df = df.dropna(subset=["y"]).copy()
        logger.info(f"Cible créée: {Counter(df['y'])} | lignes gardées: {len(df)}/{before}")
        return df

    def create_preprocessor(self):
        return ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.categorical_features),
            ("num", StandardScaler(), self.numeric_features)
        ])

    def save_correlation_matrix(self, df_engineered: pd.DataFrame):
        numeric_cols = [c for c in self.numeric_features if c in df_engineered.columns]
        if len(numeric_cols) == 0: return
        df_numeric = df_engineered[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df_numeric["y"] = df_engineered["y"]

        corr = df_numeric.corr()
        plt.figure(figsize=(12, 10))
        plt.imshow(corr, cmap="coolwarm", interpolation="none")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.title("Matrice de corrélation (features améliorées)")
        plt.tight_layout()
        plt.savefig("figures/correlation_matrix.png")
        plt.close()

        if "y" in corr.columns:
            corr_target = corr["y"].sort_values(ascending=False).drop("y", errors="ignore")
            plt.figure(figsize=(10, 6))
            corr_target.plot(kind="bar", color="skyblue")
            plt.title("Corrélations avec y")
            plt.tight_layout()
            plt.savefig("figures/correlation_with_target.png")
            plt.close()

    def save_artifacts(self, model, feature_cols: list):
        joblib.dump(model, "models/best_model.pkl")
        if self.preprocessor:
            joblib.dump(self.preprocessor, "models/preprocessor.pkl")
        with open("models/feature_cols.json", "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, ensure_ascii=False, indent=2)
        with open("models/feature_medians.json", "w", encoding="utf-8") as f:
            json.dump(self.feature_medians, f, indent=2)
        joblib.dump(self.label_encoder, "models/label_encoder.pkl")

def test_and_plot(df: pd.DataFrame):
    if "y" in df.columns:
        plt.figure(figsize=(6, 4))
        df["y"].value_counts().sort_index().plot(kind="bar")
        plt.title("Distribution de la cible (y)")
        plt.xlabel("Classe")
        plt.ylabel("N")
        plt.tight_layout()
        plt.savefig("figures/target_distribution.png")
        plt.close()

def advanced_oversample_extreme(X, y, dist):
    """Use specialized techniques for extreme imbalance"""
    try:
        # Use SMOTE instead of combined sampling for better results
        k_neighbors = min(3, dist[1] - 1)  # Very small k for tiny classes
        if k_neighbors < 1:
            # If only 1 sample, use random oversampling
            sampler = RandomOverSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            logger.info("Using RandomOverSampler for single sample class")
        else:
            # Use SMOTE for better minority class handling
            sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_res, y_res = sampler.fit_resample(X, y)
            logger.info("Using SMOTE for extreme imbalance")
    except Exception as e:
        logger.warning(f"Advanced sampling failed: {e}, using simple oversampling")
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
    
    logger.info(f"Class distribution after specialized sampling: {Counter(y_res)}")
    return X_res, y_res

# ---------------- Evaluation & plotting ----------------
def eval_and_plot(model, preprocessor, X_test_raw, y_test, name: str, tag: str, label_encoder, is_binary: bool = True):
    X_test = preprocessor.transform(X_test_raw)
    y_pred = model.predict(X_test)
    
    # Get probabilities for AUC calculation
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        if is_binary and len(label_encoder.classes_) == 2:
            auc_score = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc_score = roc_auc_score(y_test, y_proba, multi_class='ovr')
    else:
        auc_score = 0.5  # Default if no probability available

    avg = 'binary' if is_binary and len(label_encoder.classes_) == 2 else 'macro'
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average=avg, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)

    class_names = [str(cls) for cls in label_encoder.classes_]
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)

    print(f"\nClassification Report {name}:")
    print(classification_report(y_test, y_pred, target_names=class_names))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title(f"Matrice de confusion - {name} ({tag})")
    plt.tight_layout()
    plt.savefig(f"figures/confusion_matrix_{name}_{tag}.png")
    plt.close()

    # Use fixed cross-validation
    scoring_str = 'f1' if is_binary else 'f1_macro'
    try:
        cv_scores = fixed_cross_val_score(model, preprocessor.transform(X_test_raw), y_test, scoring=scoring_str)
        logger.info(f"Cross-val F1 ({name}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        cv_scores = [0]
        logger.warning(f"Cross-validation failed for {name}: {e}")

    return {"accuracy": float(accuracy), "precision": float(precision), 
            "recall": float(recall), "f1": float(f1), "auc": float(auc_score)}, report

# ---------------- Optimized Models with Enhanced Hyperparameters ----------------
def train_logreg(X_train, y_train, X_test_raw, y_test, preprocessor, label_encoder, is_binary):
    scoring_param = 'f1' if is_binary else 'f1_macro'
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100], 
        'penalty': ['l1', 'l2', 'elasticnet'],
        'solver': ['liblinear', 'saga'],
        'l1_ratio': [0.1, 0.5, 0.9]
    }
    grid = GridSearchCV(
        LogisticRegression(max_iter=10000, random_state=42, class_weight='balanced'), 
        param_grid, cv=3, scoring=scoring_param, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    metrics, report = eval_and_plot(model, preprocessor, X_test_raw, y_test, "LOGREG", "tuned", label_encoder, is_binary)
    return model, metrics, grid.best_params_, report

def train_rf(X_train, y_train, X_test_raw, y_test, preprocessor, label_encoder, is_binary):
    scoring_param = 'f1' if is_binary else 'f1_macro'
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7, 10, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced', 'balanced_subsample']
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=42), 
        param_grid, cv=3, scoring=scoring_param, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    metrics, report = eval_and_plot(model, preprocessor, X_test_raw, y_test, "RF", "tuned", label_encoder, is_binary)
    return model, metrics, grid.best_params_, report

def train_xgb(X_train, y_train, X_test_raw, y_test, preprocessor, label_encoder, is_binary):
    scoring_param = 'f1' if is_binary else 'f1_macro'
    # Calculate scale_pos_weight for imbalanced data
    scale_pos_weight = len(y_train)/(2*np.bincount(y_train)[1]) if is_binary else 1
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0],
        'scale_pos_weight': [scale_pos_weight, scale_pos_weight*2]
    }
    try:
        from xgboost import XGBClassifier
        grid = GridSearchCV(
            XGBClassifier(random_state=42, eval_metric='logloss'), 
            param_grid, cv=3, scoring=scoring_param, n_jobs=-1, verbose=1
        )
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        metrics, report = eval_and_plot(model, preprocessor, X_test_raw, y_test, "XGB", "tuned", label_encoder, is_binary)
        return model, metrics, grid.best_params_, report
    except ImportError:
        logger.warning("XGBoost not available, using GradientBoosting instead")
        return train_gb(X_train, y_train, X_test_raw, y_test, preprocessor, label_encoder, is_binary)

def train_gb(X_train, y_train, X_test_raw, y_test, preprocessor, label_encoder, is_binary):
    scoring_param = 'f1' if is_binary else 'f1_macro'
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42), 
        param_grid, cv=3, scoring=scoring_param, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    metrics, report = eval_and_plot(model, preprocessor, X_test_raw, y_test, "GB", "tuned", label_encoder, is_binary)
    return model, metrics, grid.best_params_, report

def train_svm(X_train, y_train, X_test_raw, y_test, preprocessor, label_encoder, is_binary):
    scoring_param = 'f1' if is_binary else 'f1_macro'
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto'],
        'class_weight': ['balanced']
    }
    grid = GridSearchCV(
        SVC(random_state=42, probability=True), 
        param_grid, cv=3, scoring=scoring_param, n_jobs=-1, verbose=1
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    metrics, report = eval_and_plot(model, preprocessor, X_test_raw, y_test, "SVM", "tuned", label_encoder, is_binary)
    return model, metrics, grid.best_params_, report

def optimize_threshold(model, preprocessor, X_test_raw, y_test):
    """Find optimal threshold for minority class prediction"""
    if hasattr(model, "predict_proba"):
        X_test = preprocessor.transform(X_test_raw)
        y_proba = model.predict_proba(X_test)
        
        # Try different thresholds for minority class
        best_f1 = 0
        best_threshold = 0.5
        thresholds = np.arange(0.1, 0.9, 0.05)
        for threshold in thresholds:
            y_pred = (y_proba[:, 1] >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        logger.info(f"Optimal threshold: {best_threshold}, F1: {best_f1}")
        return best_threshold
    return 0.5

def create_weighted_ensemble(models, X_train, y_train):
    """Create an ensemble weighted by model performance"""
    estimators = []
    weights = []
    
    for name, data in models.items():
        estimators.append((name, data['model']))
        weights.append(data['metrics']['f1'] + 0.1)  # Add small constant to avoid zero weights
    
    # Normalize weights
    if sum(weights) > 0:
        weights = [w/sum(weights) for w in weights]
    else:
        weights = None
    
    ensemble = VotingClassifier(
        estimators=estimators, 
        voting='soft', 
        weights=weights,
        n_jobs=-1
    )
    
    ensemble.fit(X_train, y_train)
    return ensemble

# ---------------- Main ----------------
if __name__ == "__main__":
    DEMANDES = r"C:/Users/HP/Desktop/prediction/Demandes_Conges.csv"
    SOLDE    = r"C:/Users/HP/Desktop/prediction/Solde_Conges.csv"

    pipe = CongesPipeline()

    logger.info("Loading data...")
    df_dem = load_excel_list(DEMANDES)
    df_sol = load_excel_list(SOLDE)

    logger.info("Preprocessing: Improved feature engineering...")
    df = pipe.engineer_merge(df_dem, df_sol)
    dist = Counter(df["y"])
    logger.info(f"Target distribution after engineering: {dist}")
    if len(dist) < 2:
        logger.error("Mono-class target. Fix your source labels.")
        sys.exit(1)

    logger.info("Generating plots and correlation matrices...")
    test_and_plot(df)
    pipe.save_correlation_matrix(df)

    feature_cols = pipe.categorical_features + pipe.numeric_features
    X_raw = df[feature_cols]
    y_all = df["y"].astype(int)

    # Use smaller test size to preserve more training data
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_all, test_size=0.15, random_state=42, stratify=y_all
    )

    pipe.preprocessor = pipe.create_preprocessor()
    pipe.preprocessor.fit(X_train_raw)
    X_train = pipe.preprocessor.transform(X_train_raw)
    X_test  = pipe.preprocessor.transform(X_test_raw)

    # Check for extreme class imbalance
    dist = Counter(y_train)
    logger.info(f"Training distribution: {dist}")
    
    # Handle class imbalance with advanced techniques
    X_train_res, y_train_res = advanced_oversample_extreme(X_train, y_train, dist)
        
    if len(np.unique(y_train_res)) < 2:
        logger.error("After oversampling, mono-class target. Exiting.")
        sys.exit(1)

    is_binary = len(pipe.label_encoder.classes_) == 2

    # ---------------- Train focused models ----------------
    logger.info("Training tuned Logistic Regression...")
    logreg_model, logreg_metrics, logreg_params, logreg_report = train_logreg(
        X_train_res, y_train_res, X_test_raw, y_test, pipe.preprocessor, pipe.label_encoder, is_binary
    )

    logger.info("Training tuned Random Forest...")
    rf_model, rf_metrics, rf_params, rf_report = train_rf(
        X_train_res, y_train_res, X_test_raw, y_test, pipe.preprocessor, pipe.label_encoder, is_binary
    )

    logger.info("Training XGBoost...")
    xgb_model, xgb_metrics, xgb_params, xgb_report = train_xgb(
        X_train_res, y_train_res, X_test_raw, y_test, pipe.preprocessor, pipe.label_encoder, is_binary
    )

    logger.info("Training tuned Gradient Boosting...")
    gb_model, gb_metrics, gb_params, gb_report = train_gb(
        X_train_res, y_train_res, X_test_raw, y_test, pipe.preprocessor, pipe.label_encoder, is_binary
    )

    # Create initial models dictionary
    models = {
        "logreg": {"model": logreg_model, "metrics": logreg_metrics},
        "rf": {"model": rf_model, "metrics": rf_metrics},
        "xgb": {"model": xgb_model, "metrics": xgb_metrics},
        "gb": {"model": gb_model, "metrics": gb_metrics}
    }

    # Only train additional models if we have sufficient data
    if dist[1] >= 5:  # At least 5 samples in minority class
        logger.info("Training tuned SVM...")
        svm_model, svm_metrics, svm_params, svm_report = train_svm(
            X_train_res, y_train_res, X_test_raw, y_test, pipe.preprocessor, pipe.label_encoder, is_binary
        )
        models["svm"] = {"model": svm_model, "metrics": svm_metrics}

    # ---------------- Create Weighted Ensemble ----------------
    logger.info("Creating weighted ensemble...")
    ensemble = create_weighted_ensemble(models, X_train_res, y_train_res)
    vote_metrics, vote_report = eval_and_plot(ensemble, pipe.preprocessor, X_test_raw, y_test, "ENSEMBLE", "final", pipe.label_encoder, is_binary)
    models["ensemble"] = {"model": ensemble, "metrics": vote_metrics}

    # ---------------- Select Best Model ----------------
    # Select best model based on F1 score
    best_name, best_pack = max(models.items(), key=lambda kv: kv[1]["metrics"]["f1"])
    best_model = best_pack["model"]

    # Optimize threshold for the best model
    optimal_threshold = optimize_threshold(best_model, pipe.preprocessor, X_test_raw, y_test)
    
    logger.info(f"Best model: {best_name}")
    logger.info(f"Best F1 score: {best_pack['metrics']['f1']:.4f}")
    logger.info(f"Best Accuracy: {best_pack['metrics']['accuracy']:.4f}")
    logger.info(f"Best AUC: {best_pack['metrics']['auc']:.4f}")
    logger.info(f"Optimal threshold: {optimal_threshold:.4f}")

    # Print detailed classification report
    if best_name == "ensemble":
        report = vote_report
    elif best_name == "logreg":
        report = logreg_report
    elif best_name == "rf":
        report = rf_report
    elif best_name == "xgb":
        report = xgb_report
    elif best_name == "gb":
        report = gb_report
    elif best_name == "svm":
        report = svm_report

    print(f"\n\n=== BEST MODEL: {best_name.upper()} ===")
    print(f"Accuracy: {best_pack['metrics']['accuracy']:.4f}")
    print(f"Precision: {best_pack['metrics']['precision']:.4f}")
    print(f"Recall: {best_pack['metrics']['recall']:.4f}")
    print(f"F1-score: {best_pack['metrics']['f1']:.4f}")
    print(f"AUC: {best_pack['metrics']['auc']:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print("\nClassification Report:")
    for class_name in pipe.label_encoder.classes_:
        if class_name in report:
            print(f"Class {class_name}: precision={report[class_name]['precision']:.4f}, "
                  f"recall={report[class_name]['recall']:.4f}, "
                  f"f1-score={report[class_name]['f1-score']:.4f}, "
                  f"support={report[class_name]['support']}")
    print(f"Accuracy: {report['accuracy']:.4f}")
    if 'macro avg' in report:
        print(f"Macro avg: precision={report['macro avg']['precision']:.4f}, "
              f"recall={report['macro avg']['recall']:.4f}, "
              f"f1-score={report['macro avg']['f1-score']:.4f}")
    if 'weighted avg' in report:
        print(f"Weighted avg: precision={report['weighted avg']['precision']:.4f}, "
              f"recall={report['weighted avg']['recall']:.4f}, "
              f"f1-score={report['weighted avg']['f1-score']:.4f}")

    # ---------------- Save Artifacts ----------------
    pipe.save_artifacts(best_model, feature_cols)
    with open(os.path.join("reports", f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"), "w", encoding="utf-8") as f:
        json.dump({
            "best": {"name": best_name, "metrics": best_pack["metrics"], "threshold": optimal_threshold},
            "all": {k: {"metrics": v["metrics"]} for k, v in models.items()},
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, ensure_ascii=False)

    print("Process completed at", pd.Timestamp.now())