import pandas as pd
import numpy as np
import os
import sys
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List

from src.constants import traning_pipeline
from src.entity.config_entity import DataIngestionConfig, DataValidationConfig
from src.logger.logging import logging
from src.exception.exciption import CustomException
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact

try:
    from scipy.stats import ks_2samp, chi2_contingency  # type: ignore

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

try:
    import yaml  # type: ignore

    _HAS_YAML = True
except Exception:
    _HAS_YAML = False


class DataValidation:
    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_validation_config = data_validation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def _cfg(self, name: str, default: Any) -> Any:
        """Safely read optional config attributes without breaking if not present."""
        return getattr(self.data_validation_config, name, default)

    @staticmethod
    def _psi_from_percents(
        expected_percents: np.ndarray, actual_percents: np.ndarray, eps: float = 1e-6
    ) -> float:
        expected = np.where(expected_percents <= 0, eps, expected_percents)
        actual = np.where(actual_percents <= 0, eps, actual_percents)
        return float(np.sum((actual - expected) * np.log(actual / expected)))

    def _psi_numeric(
        self, train_values: np.ndarray, test_values: np.ndarray, bins: int, eps: float
    ) -> Optional[float]:
        if train_values.size == 0 or test_values.size == 0:
            return None

        bins = max(int(bins), 2)
        quantiles = np.linspace(0.0, 1.0, bins + 1)
        edges = np.unique(np.quantile(train_values, quantiles))

        # If the feature is (almost) constant, PSI is effectively 0
        if edges.size <= 2:
            return 0.0

        # Make sure test values outside train range are still counted
        edges = edges.astype(float)
        edges[0] = -np.inf
        edges[-1] = np.inf

        expected_counts, _ = np.histogram(train_values, bins=edges)
        actual_counts, _ = np.histogram(test_values, bins=edges)

        expected_total = max(int(expected_counts.sum()), 1)
        actual_total = max(int(actual_counts.sum()), 1)

        expected_perc = expected_counts / expected_total
        actual_perc = actual_counts / actual_total
        return self._psi_from_percents(expected_perc, actual_perc, eps=eps)

    def _psi_categorical(
        self,
        train_s: pd.Series,
        test_s: pd.Series,
        max_categories: int,
        eps: float,
    ) -> Optional[float]:
        if len(train_s) == 0 or len(test_s) == 0:
            return None

        max_categories = max(int(max_categories), 2)

        train_c = train_s.fillna("__MISSING__").astype(str)
        test_c = test_s.fillna("__MISSING__").astype(str)

        combined = pd.concat([train_c, test_c], ignore_index=True)
        top_cats = combined.value_counts().head(max_categories).index

        train_c = train_c.where(train_c.isin(top_cats), "__OTHER__")
        test_c = test_c.where(test_c.isin(top_cats), "__OTHER__")

        train_dist = train_c.value_counts(normalize=True)
        test_dist = test_c.value_counts(normalize=True)

        categories = train_dist.index.union(test_dist.index)
        expected = train_dist.reindex(categories, fill_value=0).to_numpy()
        actual = test_dist.reindex(categories, fill_value=0).to_numpy()

        return self._psi_from_percents(expected, actual, eps=eps)

    @staticmethod
    def _numeric_summary(series: pd.Series) -> Dict[str, Any]:
        s = pd.to_numeric(series, errors="coerce")
        count = int(s.notna().sum())
        return {
            "count": count,
            "missing_rate": float(s.isna().mean()),
            "n_unique": int(s.nunique(dropna=True)),
            "mean": float(s.mean()) if count else None,
            "std": float(s.std(ddof=0)) if count else None,
            "min": float(s.min()) if count else None,
            "p05": float(s.quantile(0.05)) if count else None,
            "p50": float(s.quantile(0.50)) if count else None,
            "p95": float(s.quantile(0.95)) if count else None,
            "max": float(s.max()) if count else None,
        }

    @staticmethod
    def _categorical_summary(series: pd.Series, top_k: int = 10) -> Dict[str, Any]:
        top_k = max(int(top_k), 1)
        total = int(len(series))
        missing_rate = float(series.isna().mean())

        s = series.fillna("__MISSING__").astype(str)
        vc = s.value_counts(dropna=False).head(top_k)

        top_values: List[Dict[str, Any]] = []
        for val, cnt in vc.items():
            top_values.append(
                {
                    "value": str(val),
                    "count": int(cnt),
                    "pct": float(cnt / total) if total else 0.0,
                }
            )

        return {
            "count": total,
            "missing_rate": missing_rate,
            "n_unique": int(series.nunique(dropna=True)),
            "top_values": top_values,
        }

    def _build_drift_report(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Any]:
        alpha = float(self._cfg("drift_p_value_threshold", 0.05))
        psi_threshold = float(self._cfg("psi_threshold", 0.2))
        psi_bins = int(self._cfg("psi_bins", 10))
        psi_eps = float(self._cfg("psi_eps", 1e-6))
        max_categories = int(self._cfg("max_categories", 50))
        top_k = int(self._cfg("top_k", 10))

        train_cols = list(train_df.columns)
        test_cols = list(test_df.columns)

        missing_in_test = sorted(list(set(train_cols) - set(test_cols)))
        extra_in_test = sorted(list(set(test_cols) - set(train_cols)))
        common_cols = sorted(list(set(train_cols).intersection(set(test_cols))))

        # Optionally skip target column if it exists in constants
        target_col = getattr(traning_pipeline, "TARGET_COLUMN", None)
        cols_to_check = (
            [c for c in common_cols if c != target_col]
            if target_col in common_cols
            else common_cols
        )

        dtype_mismatches: Dict[str, Any] = {}
        for col in common_cols:
            if str(train_df[col].dtype) != str(test_df[col].dtype):
                dtype_mismatches[col] = {
                    "train_dtype": str(train_df[col].dtype),
                    "test_dtype": str(test_df[col].dtype),
                }

        features: Dict[str, Any] = {}
        drifted_features: List[str] = []

        for col in cols_to_check:
            train_s = train_df[col]
            test_s = test_df[col]

            is_numeric = pd.api.types.is_numeric_dtype(train_s) and pd.api.types.is_numeric_dtype(
                test_s
            )
            feature_type = "numerical" if is_numeric else "categorical"

            if feature_type == "numerical":
                train_vals = pd.to_numeric(train_s, errors="coerce").dropna().to_numpy()
                test_vals = pd.to_numeric(test_s, errors="coerce").dropna().to_numpy()

                psi_val = self._psi_numeric(
                    train_vals, test_vals, bins=psi_bins, eps=psi_eps
                )
                drifted_by_psi = (psi_val is not None) and (psi_val >= psi_threshold)

                ks_stat = None
                ks_p = None
                drifted_by_p = None
                if _HAS_SCIPY and train_vals.size > 0 and test_vals.size > 0:
                    res = ks_2samp(train_vals, test_vals)
                    ks_stat = float(res.statistic)
                    ks_p = float(res.pvalue)
                    drifted_by_p = ks_p < alpha

                drifted = drifted_by_p if drifted_by_p is not None else drifted_by_psi
                if drifted:
                    drifted_features.append(col)

                features[col] = {
                    "feature_type": feature_type,
                    "dtype": {"train": str(train_s.dtype), "test": str(test_s.dtype)},
                    "tests": {
                        "ks_2samp": {
                            "enabled": bool(_HAS_SCIPY),
                            "statistic": ks_stat,
                            "p_value": ks_p,
                            "alpha": alpha,
                        },
                        "psi": {
                            "value": psi_val,
                            "threshold": psi_threshold,
                            "bins": psi_bins,
                        },
                    },
                    "drifted": bool(drifted),
                    "drifted_by": "p_value" if drifted_by_p is not None else "psi",
                    "drifted_by_p_value": drifted_by_p,
                    "drifted_by_psi": drifted_by_psi,
                    "train_summary": self._numeric_summary(train_s),
                    "test_summary": self._numeric_summary(test_s),
                }
            else:
                psi_val = self._psi_categorical(
                    train_s, test_s, max_categories=max_categories, eps=psi_eps
                )
                drifted_by_psi = (psi_val is not None) and (psi_val >= psi_threshold)

                chi2_stat = None
                chi2_p = None
                drifted_by_p = None

                if _HAS_SCIPY:
                    train_c = train_s.fillna("__MISSING__").astype(str)
                    test_c = test_s.fillna("__MISSING__").astype(str)

                    combined = pd.concat([train_c, test_c], ignore_index=True)
                    top_cats = combined.value_counts().head(max_categories).index
                    train_c = train_c.where(train_c.isin(top_cats), "__OTHER__")
                    test_c = test_c.where(test_c.isin(top_cats), "__OTHER__")

                    train_counts = train_c.value_counts()
                    test_counts = test_c.value_counts()
                    categories = train_counts.index.union(test_counts.index)

                    if len(categories) > 1 and train_counts.sum() > 0 and test_counts.sum() > 0:
                        table = np.vstack(
                            [
                                train_counts.reindex(categories, fill_value=0).to_numpy(),
                                test_counts.reindex(categories, fill_value=0).to_numpy(),
                            ]
                        )
                        try:
                            chi2, p, _, _ = chi2_contingency(table)
                            chi2_stat = float(chi2)
                            chi2_p = float(p)
                            drifted_by_p = chi2_p < alpha
                        except Exception:
                            drifted_by_p = None

                drifted = drifted_by_p if drifted_by_p is not None else drifted_by_psi
                if drifted:
                    drifted_features.append(col)

                features[col] = {
                    "feature_type": feature_type,
                    "dtype": {"train": str(train_s.dtype), "test": str(test_s.dtype)},
                    "tests": {
                        "chi2": {
                            "enabled": bool(_HAS_SCIPY),
                            "statistic": chi2_stat,
                            "p_value": chi2_p,
                            "alpha": alpha,
                        },
                        "psi": {
                            "value": psi_val,
                            "threshold": psi_threshold,
                            "max_categories": max_categories,
                        },
                    },
                    "drifted": bool(drifted),
                    "drifted_by": "p_value" if drifted_by_p is not None else "psi",
                    "drifted_by_p_value": drifted_by_p,
                    "drifted_by_psi": drifted_by_psi,
                    "train_summary": self._categorical_summary(train_s, top_k=top_k),
                    "test_summary": self._categorical_summary(test_s, top_k=top_k),
                }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "schema_validation": {
                "train_shape": [int(train_df.shape[0]), int(train_df.shape[1])],
                "test_shape": [int(test_df.shape[0]), int(test_df.shape[1])],
                "missing_in_test": missing_in_test,
                "extra_in_test": extra_in_test,
                "dtype_mismatches": dtype_mismatches,
                "is_valid": (len(missing_in_test) == 0 and len(extra_in_test) == 0),
            },
            "drift_detection": {
                "scipy_available": bool(_HAS_SCIPY),
                "decision_rule": "p_value (KS/Chi2) if scipy is available else PSI",
                "p_value_alpha": alpha,
                "psi_threshold": psi_threshold,
                "n_features_checked": int(len(cols_to_check)),
                "n_drifted_features": int(len(drifted_features)),
                "drifted_features": drifted_features,
                "features": features,
            },
        }

    def _write_report(self, report: Dict[str, Any], file_path: str) -> None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        ext = os.path.splitext(file_path)[1].lower()

        with open(file_path, "w", encoding="utf-8") as f:
            if _HAS_YAML and ext in {".yml", ".yaml"}:
                yaml.safe_dump(report, f, sort_keys=False)  # type: ignore
            else:
                json.dump(report, f, indent=2)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Data Validation started")
            # read train and test data
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info("Data read from source completed")

            # create validated dir
            os.makedirs(self.data_validation_config.validated_dir, exist_ok=True)

            # save validated train and test data in validated dir
            validated_train_file_path = os.path.join(
                self.data_validation_config.validated_dir,
                traning_pipeline.TRAINING_FILE_NAME,
            )
            validated_test_file_path = os.path.join(
                self.data_validation_config.validated_dir, traning_pipeline.TEST_FILE_NAME
            )
            train_df.to_csv(validated_train_file_path, index=False)
            test_df.to_csv(validated_test_file_path, index=False)
            logging.info("Data saved in validated dir completed")

            # create drift report dir
            os.makedirs(self.data_validation_config.drift_report_dir, exist_ok=True)

            # save drift report in drift report dir
            drift_report_file_path = os.path.join(
                self.data_validation_config.drift_report_dir,
                traning_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
            )

            report = self._build_drift_report(train_df=train_df, test_df=test_df)
            self._write_report(report, drift_report_file_path)

            logging.info(f"Drift report saved: {drift_report_file_path}")
            logging.info(
                "Drifted features: %s/%s",
                report["drift_detection"]["n_drifted_features"],
                report["drift_detection"]["n_features_checked"],
            )

            # create data validation artifact
            data_validation_artifact = DataValidationArtifact(
                validated_train_file_path=validated_train_file_path,
                validated_test_file_path=validated_test_file_path,
                drift_report_file_path=drift_report_file_path,
            )

            return data_validation_artifact

        except Exception as e:
            raise CustomException(e, sys)