"""
Script para ETL + modelado (SMOTE + RandomForest) y exportación atómica de CSV
Limpia datos, crea features, valida multicolinealidad (VIF),
entrena pipeline con SMOTE dentro de la validación cruzada, guarda pipeline y exporta CSV listo para Power BI.

Paquetes ncesarios:
pip install pandas numpy scikit-learn imbalanced-learn statsmodels joblib

Uso sugerido CHATGPT (ejemplo):
python run_pipeline.py --input data/manufacturing_defect_dataset.csv --output_dir ./out --onedrive_dir "C:/Users/USERNAME/OneDrive/PowerBIFolder"

El script produce:
- un archivo CSV atómico: data_manu_limpia_YYYYMMDD.csv en output_dir (o en onedrive_dir si se especifica)
- un pipeline guardado: pipe_rf_smote.joblib en output_dir
- un log simple impreso en consola / archivo run.log

Notas:
- Para que Power BI refresque automáticamente, guarda el CSV en una carpeta sincronizada de OneDrive/SharePoint
  o usa un gateway si el archivo está en un servidor local.
- El pipeline incluye imputación (median), escalado, SMOTE y RandomForest.
- El cross-validation hace SMOTE dentro del pipeline para evitar data leakage.

"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import sys

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception as e:
    raise ImportError("imblearn is required. Install with: pip install imbalanced-learn") from e

try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except Exception as e:
    raise ImportError("statsmodels is required. Install with: pip install statsmodels") from e

import joblib


# ----------------------- Utility functions -----------------------

def setup_logging(output_dir: Path):
    log_path = output_dir / "run.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="a", encoding="utf-8")
        ]
    )


def detect_missing(df: pd.DataFrame) -> pd.DataFrame:
    resumen = pd.DataFrame({
        "missing_values": df.isnull().sum(),
        "blank_strings": (df == "").sum()
    })
    resumen["total_problematic"] = resumen["missing_values"] + resumen["blank_strings"]
    resumen["percent"] = (resumen["total_problematic"] / len(df)) * 100
    return resumen.sort_values("total_problematic", ascending=False)


def compute_vif(df_numeric: pd.DataFrame) -> pd.DataFrame:
    # eliminar columnas constantes
    df_var = df_numeric.loc[:, df_numeric.nunique() > 1].copy()
    cols = df_var.columns
    vif_list = []
    for i in range(df_var.shape[1]):
        try:
            vif = variance_inflation_factor(df_var.values, i)
        except Exception:
            vif = np.nan
        vif_list.append(vif)
    vif_df = pd.DataFrame({"feature": cols, "VIF": vif_list})
    return vif_df.sort_values("VIF", ascending=False)


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    # Mantener los mismos nombres que usaste y crear features seguros (evitar division por cero)
    df2 = df.copy()

    # Ejemplos: ajusta los nombres según tu dataset real
    if "ProductionVolume" in df2.columns and "ProductionCost" in df2.columns:
        df2["C_CostPerUnit"] = np.where(
            df2["ProductionVolume"] != 0,
            df2["ProductionCost"] / df2["ProductionVolume"],
            np.nan
        )

    if "ProductionVolume" in df2.columns and "EnergyConsumption" in df2.columns:
        df2["C_EnergyPerUnit"] = np.where(
            df2["ProductionVolume"] != 0,
            df2["EnergyConsumption"] / df2["ProductionVolume"],
            np.nan
        )

    if "AdditiveProcessTime" in df2.columns and "AdditiveMaterialCost" in df2.columns:
        df2["C_AdditiveCostTimeRatio"] = np.where(
            df2["AdditiveProcessTime"] != 0,
            df2["AdditiveMaterialCost"] / df2["AdditiveProcessTime"],
            np.nan
        )

    if "ProductionVolume" in df2.columns and "WorkerProductivity" in df2.columns:
        df2["C_ProductionPressure"] = df2["ProductionVolume"] * df2["WorkerProductivity"]

    if "DeliveryDelay" in df2.columns and "SupplierQuality" in df2.columns:
        df2["C_SupplyRisk"] = df2["DeliveryDelay"] * (100 - df2["SupplierQuality"])

    if "ProductionVolume" in df2.columns and "MaintenanceHours" in df2.columns:
        df2["C_MaintenanceIntensity"] = np.where(
            df2["ProductionVolume"] != 0,
            df2["MaintenanceHours"] / df2["ProductionVolume"],
            np.nan
        )

    if "DowntimePercentage" in df2.columns and "ProductionVolume" in df2.columns:
        df2["C_FailureRisk"] = df2["DowntimePercentage"] * df2["ProductionVolume"]

    if "QualityScore" in df2.columns and "DefectRate" in df2.columns and "DowntimePercentage" in df2.columns:
        df2["C_DefectRisk"] = np.where(
            df2["QualityScore"] != 0,
            (df2["DefectRate"] * df2["DowntimePercentage"]) / df2["QualityScore"],
            np.nan
        )

    if "WorkerProductivity" in df2.columns and "SafetyIncidents" in df2.columns:
        df2["C_SafetyRisk"] = np.where(
            df2["WorkerProductivity"] != 0,
            df2["SafetyIncidents"] / df2["WorkerProductivity"],
            np.nan
        )

    return df2


# ----------------------- Main pipeline -----------------------

def run_pipeline(input_path: Path, output_dir: Path, onedrive_dir: Path = None, model_path: Path = None, keep_cols: list = None):
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)
    logging.info("Starting pipeline")

    df = pd.read_csv(input_path)
    logging.info(f"Input shape: {df.shape}")

    # Feature engineering
    df_fe = feature_engineering(df)
    logging.info("Feature engineering applied")

    # Detect missing
    miss = detect_missing(df_fe)
    logging.info("Missing values summary:\n" + miss.head(10).to_string())

    # Keep columns like unique ID, line, timestamp if present
    if keep_cols is None:
        possible_keep = [c for c in ["Line","ProductionLineID","RecordTimestamp","unique_id"] if c in df_fe.columns]
    else:
        possible_keep = [c for c in keep_cols if c in df_fe.columns]

    logging.info(f"Keep cols for output: {possible_keep}")

    # TARGET
    if "DefectStatus" not in df_fe.columns:
        raise KeyError("DefectStatus column not found in input data")

    y = df_fe["DefectStatus"].copy()

    # Select numeric columns for modeling
    numeric_cols = df_fe.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "DefectStatus"]
    logging.info(f"Numeric columns for modeling ({len(numeric_cols)}): {numeric_cols}")

    X_num = df_fe[numeric_cols].copy()

    # Impute median for numeric
    imputer = SimpleImputer(strategy="median")
    X_num_imputed = pd.DataFrame(imputer.fit_transform(X_num), columns=X_num.columns, index=X_num.index)

    # VIF
    try:
        vif = compute_vif(X_num_imputed)
        logging.info("VIF top 10:\n" + vif.head(10).to_string(index=False))
        # opcional: eliminar columnas con VIF muy alto (>30)
        high_vif = vif[vif["VIF"] > 100]["feature"].tolist()
        if high_vif:
            logging.warning(f"High VIF columns detected (will be dropped): {high_vif}")
            X_num_imputed.drop(columns=high_vif, inplace=True)
    except Exception as ex:
        logging.exception("VIF computation failed; continuing without dropping features")

    # Define pipeline: imputer (again), scaler, SMOTE, model
    pipe = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
    ])

    # Cross-validation with SMOTE inside the pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    logging.info("Starting cross-validation (SMOTE inside pipeline)")

    try:
        cv_scores = cross_val_score(pipe, X_num_imputed, y, cv=cv, scoring="f1")
        logging.info(f"CV F1 scores: {cv_scores}")
        logging.info(f"CV F1 mean: {cv_scores.mean():.4f}")
    except Exception:
        logging.exception("CV failed")

    # Entrenar pipeline completo en todos los datos
    logging.info("Fitting final pipeline on all data")
    pipe.fit(X_num_imputed, y)

    # Guardar pipeline
    model_save_path = output_dir / (model_path.name if model_path is not None else "pipe_rf_smote.joblib")
    joblib.dump(pipe, model_save_path)
    logging.info(f"Pipeline saved to {model_save_path}")

    # Predecir probabilidades sobre todo el dataset (usando las mismas columnas ordenadas)
    try:
        probs = pipe.predict_proba(X_num_imputed)[:, 1]
    except Exception as ex:
        logging.exception("predict_proba failed")
        probs = np.zeros(len(df_fe))

    df_out = df_fe.copy()
    df_out["DefectProbability"] = probs
    df_out["PredictedDefect"] = (df_out["DefectProbability"] >= 0.5).astype(int)
    df_out["RiskLevel"] = pd.cut(df_out["DefectProbability"], bins=[0, 0.3, 0.6, 1], labels=["Low","Medium","High"])


    # Export columns
    export_cols = []
    # Export columns (keep stable schema) - necestio si hay muchas variballes o si queires analizar solo algunas 
    # export_cols = possible_keep + [c for c in ["ProductionVolume","DefectStatus","DefectProbability","PredictedDefect","RiskLevel","C_CostPerUnit","C_MaintenanceIntensity"] if c in df_out.columns]
    # export_cols = [c for c in export_cols if c in df_out.columns]

    if not export_cols:
        # fallback: export all columns (but stable ordering is better)
        export_cols = df_out.columns.tolist()


    # definir nombre del archivo
    fecha = datetime.now().strftime("%Y%m%d")
    filename = f"data_manu_limpia_{fecha}.csv"
    tmp_filename = filename + ".tmp"


    # decide path: prefer onedrive_dir if provisto
    if onedrive_dir is not None:
        out_path = Path(onedrive_dir) / filename
        tmp_path = Path(onedrive_dir) / tmp_filename
    else:
        out_path = output_dir / filename
        tmp_path = output_dir / tmp_filename

    # guardar de forma atómica
    try:
        df_out[export_cols].to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)
        logging.info(f"Export CSV escrito a: {out_path}")
    except Exception:
        logging.exception("Error al escribir CSV")

    logging.info("Pipeline finished successfully")
    return {
        "pipeline_path": str(model_save_path),
        "output_csv": str(out_path)
    }


# ----------------------- CLI -----------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ETL + model pipeline and export CSV for Power BI")
    parser.add_argument("--input", required=True, help="Ruta al CSV de entrada")
    parser.add_argument("--output_dir", required=True, help="Directorio donde guardar modelo y csv")
    parser.add_argument("--onedrive_dir", required=False, help="(Opcional) carpeta sincronizada de OneDrive para que Power BI haga refresh automático")
    parser.add_argument("--model_path", required=False, help="(Opcional) nombre archivo para guardar pipeline (ej: pipe_rf_smote.joblib)")
    parser.add_argument("--keep_cols", nargs="*", help="(Opcional) lista de columnas a mantener (ids, line, timestamp)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    onedrive_dir = Path(args.onedrive_dir) if args.onedrive_dir else None
    model_path = Path(args.model_path) if args.model_path else None

    res = run_pipeline(input_path=input_path, output_dir=output_dir, onedrive_dir=onedrive_dir, model_path=model_path, keep_cols=args.keep_cols)
    print("Done:", res)
