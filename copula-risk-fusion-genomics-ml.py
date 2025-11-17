import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from scipy import stats

# Optional libraries
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("xgboost not installed; XGBClassifier will be skipped.")

try:
    from lifelines import KaplanMeierFitter
    HAS_LIFELINES = True
except ImportError:
    HAS_LIFELINES = False
    print("lifelines not installed; Kaplan–Meier curves will be skipped.")


# ------------------------------------------------------------------------
# Basic utilities
# ------------------------------------------------------------------------

def load_metabric(csv_path="METABRIC_RNA_Mutation.csv"):
    df = pd.read_csv(csv_path)
    print(f"Loaded METABRIC with shape {df.shape}")
    return df


def build_5year_endpoint(df,
                         time_col="overall_survival_months",
                         status_col_overall="overall_survival",
                         status_col_cancer="death_from_cancer",
                         cutoff_months=60.0):
    """
    Build binary endpoint: cancer death within cutoff_months.

    Logic:
    - Prefer cause-specific 'death_from_cancer' if available.
    - Otherwise fall back to 'overall_survival'.
    - Handle numeric 0/1 and string labels like 'Died of Disease', 'Living', etc.
    - Subjects censored before cutoff (time < cutoff and status==0) are set to NaN and dropped later.
    """

    time = df[time_col].astype(float)

    # Choose status column
    if status_col_cancer in df.columns:
        raw = df[status_col_cancer]
        print("Using 'death_from_cancer' for event status.")
    else:
        raw = df[status_col_overall]
        print("Using 'overall_survival' for event status (no death_from_cancer column).")

    # Parse status
    if raw.dtype.kind in "ifb":
        status = (raw.astype(int) == 1).astype(int)
    else:
        s = raw.astype(str).str.strip().str.lower()
        died_labels = {
            "died of disease",
            "died",
            "dead",
            "deceased",
            "1",
            "yes",
            "true"
        }
        alive_labels = {
            "living",
            "alive",
            "0",
            "no",
            "false"
        }
        status = np.where(s.isin(died_labels), 1,
                          np.where(s.isin(alive_labels), 0, np.nan))
        status = pd.Series(status, index=df.index)

    # event_5y definition
    event_5y = pd.Series(np.nan, index=df.index, dtype=float)

    mask_event = (status == 1) & (time <= cutoff_months)
    event_5y.loc[mask_event] = 1.0

    mask_alive_known = ((status == 1) & (time > cutoff_months)) | ((status == 0) & (time >= cutoff_months))
    event_5y.loc[mask_alive_known] = 0.0

    print(f"5-year endpoint: {int((event_5y == 1).sum())} events / "
          f"{int((event_5y == 0).sum())} non-events / "
          f"{int(event_5y.isna().sum())} censored<5y (dropped)")
    return event_5y


def split_views(df):
    """
    Split into:
        - clinical_cols: the 31 clinical variables as defined in Kaggle description
        - genomic_cols: all remaining columns (expression + mutation), excluding IDs and survival

    This avoids fragile heuristics and ensures we always get a non-empty genomic view.
    """
    clinical_candidates = [
        "patient_id",
        "age_at_diagnosis",
        "type_of_breast_surgery",
        "cancer_type",
        "cancer_type_detailed",
        "cellularity",
        "chemotherapy",
        "pam50_+_claudin-low_subtype",
        "cohort",
        "er_status_measured_by_ihc",
        "er_status",
        "neoplasm_histologic_grade",
        "her2_status_measured_by_snp6",
        "her2_status",
        "tumor_other_histologic_subtype",
        "hormone_therapy",
        "inferred_menopausal_state",
        "integrative_cluster",
        "primary_tumor_laterality",
        "lymph_nodes_examined_positive",
        "mutation_count",
        "nottingham_prognostic_index",
        "oncotree_code",
        "overall_survival_months",
        "overall_survival",
        "pr_status",
        "radio_therapy",
        "3-gene_classifier_subtype",
        "tumor_size",
        "tumor_stage",
        "death_from_cancer"
    ]

    # Keep only those that actually exist
    clinical_cols = [c for c in clinical_candidates if c in df.columns]

    id_cols = {"patient_id"}
    survival_cols = {"overall_survival_months", "overall_survival", "death_from_cancer"}

    # Genomic = everything not in clinical, id, survival
    genomic_cols = [c for c in df.columns
                    if c not in clinical_cols
                    and c not in id_cols
                    and c not in survival_cols]

    # For downstream convenience, we also define mut_cols as those ending with '_mut'
    mut_cols = [c for c in genomic_cols if c.endswith("_mut")]

    # Remove mut_cols from "pure" expression if you ever want to separate
    print(f"Clinical cols: {len(clinical_cols)}, genomic cols: {len(genomic_cols)}, mut cols: {len(mut_cols)}")
    return clinical_cols, genomic_cols, mut_cols


def build_ml_view(df, cols, y, view_name, random_state=42):
    """
    Build ML classifiers on a given view.
    Returns:
        - best_model_name
        - best_model (fitted)
        - cv_risk_scores (CV predicted probabilities)
        - metrics dict
    """
    X = df[cols].copy()
    print(f"[{view_name}] Using {X.shape[1]} features.")

    num_cols = [c for c in cols if df[c].dtype.kind in "if"]
    cat_cols = [c for c in cols if c not in num_cols]

    X[num_cols] = X[num_cols].fillna(X[num_cols].median())
    for c in cat_cols:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].mode().iloc[0])

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", numeric_transformer, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", categorical_transformer, cat_cols))

    if len(transformers) == 0:
        raise ValueError(f"[{view_name}] No usable features found (neither numeric nor categorical).")

    preprocessor = ColumnTransformer(transformers=transformers)

    estimators = []

    # Logistic Regression (elastic net)
    log_reg = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.5,
        max_iter=5000,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state
    )
    estimators.append(("logistic_en", log_reg))

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=5,
        n_jobs=-1,
        class_weight="balanced",
        random_state=random_state
    )
    estimators.append(("random_forest", rf))

    # Gradient Boosting
    gb = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=3,
        random_state=random_state
    )
    estimators.append(("grad_boost", gb))

    # XGBoost (if available)
    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=random_state
        )
        estimators.append(("xgboost", xgb))

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    metrics = {}
    best_auc = -np.inf
    best_name = None
    best_model = None
    best_cv_scores = None

    for name, clf in estimators:
        pipe = Pipeline(steps=[("pre", preprocessor),
                               ("clf", clf)])
        print(f"\n[{view_name}] Training model: {name}")
        cv_probs = cross_val_predict(
            pipe, X, y, cv=skf,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]
        auc_score = roc_auc_score(y, cv_probs)
        print(f"[{view_name}] {name} CV ROC AUC: {auc_score:.3f}")
        metrics[name] = auc_score

        if auc_score > best_auc:
            best_auc = auc_score
            best_name = name
            best_cv_scores = cv_probs
            pipe.fit(X, y)
            best_model = pipe

    print(f"\n[{view_name}] Best model: {best_name} (ROC AUC = {best_auc:.3f})")

    return best_name, best_model, best_cv_scores, metrics


# ------------------------------------------------------------------------
# Copula functions
# ------------------------------------------------------------------------

EPS = 1e-10


def empirical_copula_C(u, v):
    n = len(u)
    Cn = np.empty(n)
    for i in range(n):
        Cn[i] = np.mean((u <= u[i]) & (v <= v[i]))
    return Cn


# ---- Gaussian copula ----

def copula_gaussian_cdf(u, v, rho):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    cov = [[1, rho], [rho, 1]]
    return stats.multivariate_normal.cdf(np.column_stack([x, y]),
                                         mean=[0, 0], cov=cov)


def copula_gaussian_pdf(u, v, rho):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    cov = np.array([[1, rho], [rho, 1]])
    inv_cov = np.linalg.inv(cov)
    det_cov = np.linalg.det(cov)
    xy = np.column_stack([x, y])
    quad = np.einsum("ij,jk,ik->i", xy, inv_cov, xy)
    num = np.exp(-0.5 * quad)
    den = 2 * np.pi * np.sqrt(det_cov)
    pdf_xy = num / den
    pdf_x = stats.norm.pdf(x)
    pdf_y = stats.norm.pdf(y)
    return pdf_xy / (pdf_x * pdf_y)


def fit_gaussian_copula(u, v):
    tau = stats.kendalltau(u, v)[0]
    rho = np.sin(np.pi * tau / 2.0)
    return rho


def simulate_gaussian_copula(n, rho, random_state=None):
    rng = np.random.default_rng(random_state)
    cov = np.array([[1, rho], [rho, 1]])
    x = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n)
    u = stats.norm.cdf(x[:, 0])
    v = stats.norm.cdf(x[:, 1])
    return u, v


# ---- Clayton copula ----

def copula_clayton_cdf(u, v, theta):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    return np.maximum((u ** (-theta) + v ** (-theta) - 1.0) ** (-1.0 / theta), EPS)


def copula_clayton_pdf(u, v, theta):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    t1 = (theta + 1.0) * (u * v) ** (-(theta + 1.0))
    t2 = (u ** (-theta) + v ** (-theta) - 1.0) ** (-1.0 / theta - 2.0)
    return t1 * t2


def fit_clayton_copula(u, v):
    tau = stats.kendalltau(u, v)[0]
    if tau <= 0:
        tau = 1e-6
    theta = 2 * tau / (1 - tau)
    return theta


# ---- Gumbel copula ----

def copula_gumbel_cdf(u, v, theta):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    t = ((-np.log(u)) ** theta + (-np.log(v)) ** theta) ** (1.0 / theta)
    return np.exp(-t)


def copula_gumbel_pdf(u, v, theta):
    u = np.clip(u, EPS, 1 - EPS)
    v = np.clip(v, EPS, 1 - EPS)
    log_u = -np.log(u)
    log_v = -np.log(v)
    t_theta = log_u ** theta + log_v ** theta
    t = t_theta ** (1.0 / theta)
    c = np.exp(-t) * (log_u * log_v) ** (theta - 1.0)
    c *= (1 + (theta - 1.0) * t_theta ** (-1.0 / theta)) / (u * v * t_theta ** (2.0 - 2.0 / theta))
    return np.clip(c, EPS, None)


def fit_gumbel_copula(u, v):
    tau = stats.kendalltau(u, v)[0]
    if tau < 0:
        tau = 0.0
    theta = 1.0 / (1.0 - tau + 1e-6)
    theta = max(theta, 1.0)
    return theta


# ---- Tail dependence ----

def tail_dependence_gaussian(rho):
    if abs(rho) >= 1.0 - 1e-6:
        return 1.0, 1.0
    return 0.0, 0.0


def tail_dependence_clayton(theta):
    if theta <= 0:
        return 0.0, 0.0
    lambda_L = 2 ** (-1.0 / theta)
    lambda_U = 0.0
    return lambda_L, lambda_U


def tail_dependence_gumbel(theta):
    if theta < 1:
        theta = 1
    lambda_U = 2 - 2 ** (1.0 / theta)
    lambda_L = 0.0
    return lambda_L, lambda_U


# ------------------------------------------------------------------------
# Archimedean simulation via Algorithm I (Genest & Rivest, 1993)
# ------------------------------------------------------------------------

def _simulate_archimedean_alg1(n, theta, family, random_state=None):
    """
    Algorithm I (Genest & Rivest, 1993) for Archimedean copulas.

    1. Generate s, t ~ U(0,1).
    2. Compute w = K^{-1}(t), where K(x) = x - phi(x)/phi'(x).
       We approximate K^{-1} via monotone interpolation on a fine grid.
    3. Set u = phi^{-1}( s * phi(w) ), v = phi^{-1}( (1-s) * phi(w) ).

    Implemented for:
        - family='clayton'
        - family='gumbel'
    """
    rng = np.random.default_rng(random_state)
    s = rng.uniform(size=n)
    t = rng.uniform(size=n)
    eps = 1e-10

    if family.lower() == "clayton":
        # Generator: phi(t) = (t^{-theta} - 1)/theta
        def phi(x):
            return (x ** (-theta) - 1.0) / theta

        def phi_inv(w):
            return (1.0 + theta * w) ** (-1.0 / theta)

        def phi_prime(x):
            # d/dx phi(x) = - x^{-(theta+1)}
            return -x ** (-(theta + 1.0))

    elif family.lower() == "gumbel":
        # Generator: phi(t) = (-log t)^theta
        def phi(x):
            return (-np.log(x)) ** theta

        def phi_inv(w):
            return np.exp(-w ** (1.0 / theta))

        def phi_prime(x):
            # d/dx phi(x) = theta * (-log x)^{theta-1} * (-1/x)
            return -theta * (-np.log(x)) ** (theta - 1.0) / x
    else:
        raise ValueError(f"Unsupported Archimedean family: {family}")

    def K(x):
        return x - phi(x) / phi_prime(x)

    # Precompute K(x) on a fine grid and invert by interpolation
    x_grid = np.linspace(1e-6, 1 - 1e-6, 5000)
    K_grid = K(x_grid)

    # Ensure monotone increasing for interpolation
    idx = np.argsort(K_grid)
    K_sorted = K_grid[idx]
    x_sorted = x_grid[idx]

    K_min, K_max = K_sorted[0], K_sorted[-1]
    t_clipped = np.clip(t, K_min, K_max)
    w = np.interp(t_clipped, K_sorted, x_sorted)
    w = np.clip(w, eps, 1 - eps)

    phi_w = phi(w)
    u = phi_inv(s * phi_w)
    v = phi_inv((1.0 - s) * phi_w)

    u = np.clip(u, eps, 1 - eps)
    v = np.clip(v, eps, 1 - eps)
    return u, v


def simulate_clayton_copula(n, theta, random_state=None):
    """
    Clayton copula simulation via Algorithm I (Genest & Rivest, 1993)
    using the Clayton generator phi(t) = (t^{-theta} - 1)/theta.
    """
    return _simulate_archimedean_alg1(n, theta, family="clayton", random_state=random_state)


def simulate_gumbel_copula(n, theta, random_state=None):
    """
    Gumbel copula simulation via Algorithm I (Genest & Rivest, 1993)
    using the Gumbel generator phi(t) = (-log t)^theta.
    """
    return _simulate_archimedean_alg1(n, theta, family="gumbel", random_state=random_state)


# ------------------------------------------------------------------------
# Cramer–von Mises GOF for copulas with bootstrap
# ------------------------------------------------------------------------

def cvm_statistic(u, v, copula_cdf, params):
    Cn = empirical_copula_C(u, v)
    Ct = copula_cdf(u, v, *params)
    return np.mean((Cn - Ct) ** 2)


def cvm_gof_bootstrap(u, v, copula_cdf, copula_sim, params,
                      n_bootstrap=1000, random_state=42):
    rng = np.random.default_rng(random_state)
    obs_stat = cvm_statistic(u, v, copula_cdf, params)

    boot_stats = []
    n = len(u)
    for b in range(n_bootstrap):
        u_b, v_b = copula_sim(n, *params, random_state=rng.integers(1e9))
        stat_b = cvm_statistic(u_b, v_b, copula_cdf, params)
        boot_stats.append(stat_b)

    boot_stats = np.array(boot_stats)
    p_value = (1.0 + np.sum(boot_stats >= obs_stat)) / (n_bootstrap + 1.0)
    return obs_stat, p_value, boot_stats


# ------------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------------

def plot_histograms(risk_clin, risk_gen, outdir):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(risk_clin, bins=30, alpha=0.7)
    plt.title("Clinical risk score distribution")
    plt.xlabel("Risk score")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.hist(risk_gen, bins=30, alpha=0.7)
    plt.title("Genomic risk score distribution")
    plt.xlabel("Risk score")
    plt.tight_layout()
    plt.savefig(outdir / "risk_histograms.png", dpi=300)
    plt.close()


def plot_risk_scatter(risk_clin, risk_gen, y, outdir):
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(risk_clin, risk_gen, c=y, cmap="coolwarm", alpha=0.7)
    plt.colorbar(sc, label="5-year cancer death (1=yes,0=no)")
    plt.xlabel("Clinical risk score")
    plt.ylabel("Genomic risk score")
    plt.title("Clinical vs Genomic risk")
    plt.tight_layout()
    plt.savefig(outdir / "risk_scatter.png", dpi=300)
    plt.close()


def plot_roc_curves(y, scores_dict, outdir, title="ROC curves"):
    plt.figure(figsize=(6, 6))
    for label, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(outdir / "roc_curves.png", dpi=300)
    plt.close()


def plot_km_joint_risk(time, status, risk_clin, risk_gen, outdir):
    if not HAS_LIFELINES:
        return

    hc = risk_clin > np.median(risk_clin)
    hg = risk_gen > np.median(risk_gen)

    groups = {
        "low-low": (~hc) & (~hg),
        "high-clinical-only": hc & (~hg),
        "high-genomic-only": (~hc) & hg,
        "high-both": hc & hg
    }

    kmf = KaplanMeierFitter()
    plt.figure(figsize=(7, 6))

    for label, mask in groups.items():
        if mask.sum() < 10:
            continue
        kmf.fit(time[mask], event_observed=status[mask], label=label)
        kmf.plot(ci_show=False)

    plt.xlabel("Time (months)")
    plt.ylabel("Survival probability")
    plt.title("Kaplan–Meier by joint risk strata")
    plt.tight_layout()
    plt.savefig(outdir / "km_joint_risk.png", dpi=300)
    plt.close()


def plot_copula_contours(u, v, copula_cdf, params, outdir, name):
    plt.figure(figsize=(6, 6))
    plt.scatter(u, v, alpha=0.4, s=10)
    grid = np.linspace(0.01, 0.99, 50)
    U, V = np.meshgrid(grid, grid)
    C = copula_cdf(U.ravel(), V.ravel(), *params).reshape(U.shape)
    cs = plt.contour(U, V, C, levels=10, colors="k", linewidths=0.5)
    plt.clabel(cs, inline=1, fontsize=8)
    plt.xlabel("U (clinical risk rank)")
    plt.ylabel("V (genomic risk rank)")
    plt.title(f"{name} copula contours on pseudo-observations")
    plt.tight_layout()
    plt.savefig(outdir / f"copula_{name}_contours.png", dpi=300)
    plt.close()


def plot_copula_heatmap(u, v, copula_cdf, params, outdir, name):
    grid = np.linspace(0.01, 0.99, 30)
    U, V = np.meshgrid(grid, grid)

    emp_vals = []
    for i in range(len(grid)):
        for j in range(len(grid)):
            emp_vals.append(np.mean((u <= U[i, j]) & (v <= V[i, j])))
    emp_vals = np.array(emp_vals).reshape(U.shape)

    fit_vals = copula_cdf(U.ravel(), V.ravel(), *params).reshape(U.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    im0 = axes[0].imshow(emp_vals, origin="lower",
                         extent=[0, 1, 0, 1], aspect="auto")
    axes[0].set_title("Empirical copula")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("v")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(fit_vals, origin="lower",
                         extent=[0, 1, 0, 1], aspect="auto")
    axes[1].set_title(f"Fitted {name} copula")
    axes[1].set_xlabel("u")
    axes[1].set_ylabel("v")
    fig.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(outdir / f"copula_{name}_emp_vs_fit.png", dpi=300)
    plt.close()


# ------------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------------

def main():
    outdir = Path("study1_outputs")
    outdir.mkdir(exist_ok=True)

    df = load_metabric()
    y_5y = build_5year_endpoint(df)

    # Drop censored<5y (NaN endpoint)
    mask_valid = ~y_5y.isna()
    df = df.loc[mask_valid].reset_index(drop=True)
    y = y_5y.loc[mask_valid].astype(int).reset_index(drop=True)
    print(f"Analysis dataset after dropping censored<5y: {df.shape}, "
          f"events={int(y.sum())}, non-events={int((1 - y).sum())}")

    clinical_cols, genomic_cols, mut_cols = split_views(df)


    # Exclude survival / outcome columns from the clinical predictors
    survival_cols = ["overall_survival_months", "overall_survival", "death_from_cancer"]
    clinical_ml_cols = [c for c in clinical_cols if c not in survival_cols]

    print(f"Clinical ML features (after removing survival columns): {len(clinical_ml_cols)}")

    # Clinical ML view (no leakage now)
    best_name_clin, best_model_clin, cv_risk_clin, metrics_clin = build_ml_view(
        df, clinical_ml_cols, y, view_name="clinical", random_state=42
    )
    # ----------------------------------------------------

    # Genomic ML view: we’ll still pick top 50 by variance from genomic_cols
    genomic_df = df[genomic_cols].select_dtypes(include=[np.number])
    variances = genomic_df.var().sort_values(ascending=False)
    top_genes = variances.head(50).index.tolist()
    print(f"Using top {len(top_genes)} genomic numeric features by variance for genomic view.")
    best_name_gen, best_model_gen, cv_risk_gen, metrics_gen = build_ml_view(
        df, top_genes, y, view_name="genomic", random_state=42
    )

    perf_df = pd.DataFrame({
        "model": list(metrics_clin.keys()) + [f"genomic_{k}" for k in metrics_gen.keys()],
        "view": ["clinical"] * len(metrics_clin) + ["genomic"] * len(metrics_gen),
        "auc": list(metrics_clin.values()) + list(metrics_gen.values())
    })
    perf_df.to_csv(outdir / "model_performance.csv", index=False)

    risk_clin = cv_risk_clin
    risk_gen = cv_risk_gen

    risk_df = pd.DataFrame({
        "clinical_risk": risk_clin,
        "genomic_risk": risk_gen,
        "y_5y": y
    })
    risk_df.to_csv(outdir / "risk_scores.csv", index=False)

    plot_histograms(risk_clin, risk_gen, outdir)
    plot_risk_scatter(risk_clin, risk_gen, y, outdir)
    plot_roc_curves(y, {"clinical": risk_clin, "genomic": risk_gen},
                    outdir, title="Clinical vs Genomic risk ROC")

    # KM curves using original follow-up times / status
    time = df["overall_survival_months"].astype(float)
    if "death_from_cancer" in df.columns:
        raw_status = df["death_from_cancer"]
    else:
        raw_status = df["overall_survival"]

    if raw_status.dtype.kind in "ifb":
        status_surv = (raw_status.astype(int) == 1).astype(int).values
    else:
        s = raw_status.astype(str).str.strip().str.lower()
        died_labels = {"died of disease", "died", "dead", "deceased", "1", "yes", "true"}
        status_surv = s.isin(died_labels).astype(int).values

    plot_km_joint_risk(time.values, status_surv, risk_clin, risk_gen, outdir)

    # --------------------------------------------------------------------
    # Copula analysis
    # --------------------------------------------------------------------
    n = len(risk_clin)
    rank_clin = stats.rankdata(risk_clin, method="average")
    rank_gen = stats.rankdata(risk_gen, method="average")
    u = rank_clin / (n + 1.0)
    v = rank_gen / (n + 1.0)

    print("\nFitting copulas on pseudo-observations...")

    rho_gauss = fit_gaussian_copula(u, v)
    lamL_gauss, lamU_gauss = tail_dependence_gaussian(rho_gauss)
    print(f"Gaussian copula: rho={rho_gauss:.3f}, lambda_L={lamL_gauss:.3f}, lambda_U={lamU_gauss:.3f}")

    theta_clay = fit_clayton_copula(u, v)
    lamL_clay, lamU_clay = tail_dependence_clayton(theta_clay)
    print(f"Clayton copula: theta={theta_clay:.3f}, lambda_L={lamL_clay:.3f}, lambda_U={lamU_clay:.3f}")

    theta_gum = fit_gumbel_copula(u, v)
    lamL_gum, lamU_gum = tail_dependence_gumbel(theta_gum)
    print(f"Gumbel copula: theta={theta_gum:.3f}, lambda_L={lamL_gum:.3f}, lambda_U={lamU_gum:.3f}")

    tail_df = pd.DataFrame([
        {"copula": "Gaussian", "param": rho_gauss, "lambda_L": lamL_gauss, "lambda_U": lamU_gauss},
        {"copula": "Clayton",  "param": theta_clay, "lambda_L": lamL_clay, "lambda_U": lamU_clay},
        {"copula": "Gumbel",   "param": theta_gum,  "lambda_L": lamL_gum, "lambda_U": lamU_gum},
    ])
    tail_df.to_csv(outdir / "copula_tail_dependence.csv", index=False)

    print("\nCramér–von Mises GOF with bootstrap (1000 reps per copula)...")

    gof_results = []

    stat_g, p_g, boot_g = cvm_gof_bootstrap(
        u, v, copula_gaussian_cdf, simulate_gaussian_copula,
        params=(rho_gauss,), n_bootstrap=1000, random_state=42
    )
    gof_results.append({"copula": "Gaussian", "stat": stat_g, "p_value": p_g})

    stat_c, p_c, boot_c = cvm_gof_bootstrap(
        u, v, copula_clayton_cdf, simulate_clayton_copula,
        params=(theta_clay,), n_bootstrap=1000, random_state=43
    )
    gof_results.append({"copula": "Clayton", "stat": stat_c, "p_value": p_c})

    stat_gu, p_gu, boot_gu = cvm_gof_bootstrap(
        u, v, copula_gumbel_cdf, simulate_gumbel_copula,
        params=(theta_gum,), n_bootstrap=1000, random_state=44
    )
    gof_results.append({"copula": "Gumbel", "stat": stat_gu, "p_value": p_gu})

    gof_df = pd.DataFrame(gof_results)
    gof_df.to_csv(outdir / "copula_gof_cvm.csv", index=False)
    print(gof_df)

    # Choose copula with highest p-value
    best_idx = gof_df["p_value"].idxmax()
    best_copula_name = gof_df.loc[best_idx, "copula"]
    print(f"\nBest-fitting copula by GOF p-value: {best_copula_name}")

    if best_copula_name == "Gaussian":
        params = (rho_gauss,)
        cdf_func = copula_gaussian_cdf
    elif best_copula_name == "Clayton":
        params = (theta_clay,)
        cdf_func = copula_clayton_cdf
    else:
        params = (theta_gum,)
        cdf_func = copula_gumbel_cdf

    plot_copula_contours(u, v, cdf_func, params, outdir, best_copula_name)
    plot_copula_heatmap(u, v, cdf_func, params, outdir, best_copula_name)

    print("\nStudy 1 pipeline complete. Outputs saved in 'study1_outputs' directory.")


if __name__ == "__main__":
    main()
