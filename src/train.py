from src.config import *
from src.loader import *
from src.wildfire import *
from src.feature import *
from src.preprocessing import *
from src.classical import *
from src.quantum import (
    load_task1, run_qrc, run_qrc_regression,
    run_trainable_kernel, run_qlstm, run_transfer_learning,
)
from src.clustering import *
from src.visualization import *
from src.insurance_modeling import run_insurance, load_insurance, build_insurance_sequences
from src.geospatial import *


def run():
    # ── Classical pipeline (wildfire_county_monthly) ──────────────────────────
    df = load_csv(DATA_PATHS["wildfire"])
    df = prepare_wildfire(df)
    df = create_features(df)

    X_train, X_test, y_train, y_test = split_scale(
        df, TARGET_COL, TEST_SIZE, RANDOM_STATE
    )

    rf = train_rf(X_train, y_train)
    evaluate(rf, X_test, y_test)

    # Clustering
    emb, labels = run_clustering(X_train)
    plot_clusters(emb, labels)

    # ── Quantum pipeline (wildfire_weather_daily) ─────────────────────────────
    df_daily = load_csv(DATA_PATHS["wildfire_daily"])
    X_tr_q, X_te_q, y_tr, y_te, X_tr_s, X_te_s = load_task1(df_daily)

    # A — QRC classification
    f1_qrc, auc_qrc, _, _, feat_tr, feat_te, y_tr_sub, y_te_sub = run_qrc(
        X_tr_q, X_te_q, y_tr, y_te
    )

    # B — Trainable kernel
    f1_kernel, auc_kernel, _, _ = run_trainable_kernel(X_tr_q, X_te_q, y_tr, y_te)

    # D — Transfer learning
    f1_tl, auc_tl, _, _ = run_transfer_learning(X_tr_s, X_te_s, y_tr, y_te)

    # ── Insurance (Task 2) ────────────────────────────────────────────────────
    model_ins, ins, col_prem, col_exp, col_risk = run_insurance(
        DATA_PATHS["insurance_2018"], DATA_PATHS["insurance_2020"]
    )

    # C — QLSTM on insurance sequences
    feature_cols = [col_risk, col_exp, "total_insured_loss"]
    sequences, targets = build_insurance_sequences(ins, col_prem, feature_cols)
    qlstm_preds, y_te_ins = run_qlstm(sequences, targets)

    # ── Geo ───────────────────────────────────────────────────────────────────
    plot_map(DATA_PATHS["geo"])

    print("\n=== Summary ===")
    print(f"RF Accuracy:            see above")
    print(f"QRC F1:                 {f1_qrc:.4f}  AUC: {auc_qrc:.4f}")
    print(f"Trainable Kernel F1:    {f1_kernel:.4f}  AUC: {auc_kernel:.4f}")
    print(f"Transfer Learning F1:   {f1_tl:.4f}  AUC: {auc_tl:.4f}")
