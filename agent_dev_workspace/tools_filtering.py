"""Tools for feature selection based on statistical filtering (variance, correlation)."""

from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import sparse, stats
from langchain_core.tools import tool


def make_tools(session):
    """Factory function that returns all filtering tools bound to session."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> Dict[str, Any]:
        """Filter DTM features by variance to remove near-constant terms.

        Computes the population variance for each term across all documents in
        session.feature_matrix. Terms with variance below the specified threshold
        are flagged for removal. Stores a DataFrame full_report formatted for
        visualize_result_tool (sorted descending by variance).

        Args:
            threshold: Minimum variance required to keep a feature (default 0.0).

        Returns:
            Dict containing summary statistics (total terms, terms kept/removed,
            variance threshold, top terms) and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix (DTM) found. Please run build_dtm_tool first."
            }

        X = session.feature_matrix
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        n_docs = X.shape[0]
        if n_docs == 0:
            return {"error": "DTM has 0 documents."}

        # Sparse population variance calculation: Var(X) = E[X^2] - (E[X])^2
        # E[X] (mean per column)
        mean = np.ravel(X.mean(axis=0))
        # E[X^2] (mean of squared elements per column)
        X_squared = X.copy()
        X_squared.data **= 2
        mean_sq = np.ravel(X_squared.mean(axis=0))

        variances = mean_sq - (mean ** 2)

        feature_names = list(session.feature_names)
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        })

        # Sort descending by variance for visualization compatibility
        df_report = df_report.sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_df = df_report[df_report["variance"] >= threshold]
        removed_df = df_report[df_report["variance"] < threshold]

        result_id = session.next_result_id("variance_filter")

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "terms_kept": len(kept_df),
            "terms_removed": len(removed_df),
            "top_terms": df_report.head(10).to_dict(orient="records")
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str) -> Dict[str, Any]:
        """Compute Pearson correlation between each term's count and a binary target indicator.

        Creates a 1-vs-rest binary indicator vector for target_class using session.labels
        (or session.dataframe['category_name']) and calculates Pearson correlation (pearson_r)
        for each term in session.feature_matrix using vectorized sparse matrix operations.
        Stores a DataFrame full_report formatted for visualize_result_tool (sorted by absolute
        correlation descending).

        Args:
            target_class: Label string of the target category to correlate against.

        Returns:
            Dict containing summary statistics (target_class, total_terms, top_terms)
            and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix (DTM) found. Please run build_dtm_tool first."
            }

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].to_numpy()

        if labels is None:
            return {"error": "No category labels found in session."}

        # Create 1-vs-rest binary target indicator y (1 for target_class, 0 otherwise)
        y = (np.array(labels) == target_class).astype(float)
        if y.sum() == 0:
            return {"error": f"Target class '{target_class}' not found in labels."}

        X = session.feature_matrix
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        n_docs = X.shape[0]
        if n_docs == 0:
            return {"error": "DTM has 0 documents."}

        # Vectorized Pearson correlation over sparse X: r = Cov(X, y) / (std_X * std_y)
        y_mean = y.mean()
        y_std = y.std(ddof=0)

        if y_std == 0:
            pearson_r = np.zeros(X.shape[1])
        else:
            # E[X] and std(X) across columns
            x_mean = np.ravel(X.mean(axis=0))
            
            X_sq = X.copy()
            X_sq.data **= 2
            x_mean_sq = np.ravel(X_sq.mean(axis=0))
            x_var = x_mean_sq - (x_mean ** 2)
            # Clip small negative values due to floating point precision before sqrt
            x_std = np.sqrt(np.maximum(0.0, x_var))

            # E[X * y] via sparse matrix-vector dot product X.T @ y / n_docs
            xy_mean = np.ravel(X.T.dot(y)) / n_docs
            cov_xy = xy_mean - (x_mean * y_mean)

            # Avoid division by zero when term standard deviation is zero
            with np.errstate(divide='ignore', invalid='ignore'):
                pearson_r = np.where(x_std > 0, cov_xy / (x_std * y_std), 0.0)

        feature_names = list(session.feature_names)
        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": pearson_r
        })

        # Sort by absolute pearson_r descending for visualization compatibility
        df_report["abs_r"] = df_report["pearson_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms": len(feature_names),
            "top_terms": df_report.head(10).to_dict(orient="records")
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, max_terms: int = 1500) -> Dict[str, Any]:
        """Compute Spearman rank correlation between term counts and a target class.

        Filters DTM terms to the top max_terms by variance first (to avoid dense memory bottlenecks),
        then computes Spearman rank correlation (spearman_r) for each selected term against a 1-vs-rest
        binary indicator vector for target_class. Stores a DataFrame full_report formatted for
        visualize_result_tool (sorted by absolute correlation descending).

        Args:
            target_class: Label string of the target category to correlate against.
            max_terms: Maximum number of top-variance terms to compute rank correlation for (default 1500).

        Returns:
            Dict containing summary statistics (target_class, total_terms, evaluated_terms, top_terms)
            and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No Document-Term Matrix (DTM) found. Please run build_dtm_tool first."
            }

        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].to_numpy()

        if labels is None:
            return {"error": "No category labels found in session."}

        # Create 1-vs-rest binary target indicator y (1 for target_class, 0 otherwise)
        y = (np.array(labels) == target_class).astype(float)
        if y.sum() == 0:
            return {"error": f"Target class '{target_class}' not found in labels."}

        X = session.feature_matrix
        if not sparse.issparse(X):
            X = sparse.csr_matrix(X)

        n_docs, n_terms = X.shape
        if n_docs == 0:
            return {"error": "DTM has 0 documents."}

        feature_names = np.array(session.feature_names)

        # Select top max_terms terms by variance to limit dense ranking overhead
        x_mean = np.ravel(X.mean(axis=0))
        X_sq = X.copy()
        X_sq.data **= 2
        x_mean_sq = np.ravel(X_sq.mean(axis=0))
        variances = x_mean_sq - (x_mean ** 2)

        if n_terms > max_terms:
            top_indices = np.argsort(variances)[::-1][:max_terms]
        else:
            top_indices = np.arange(n_terms)

        X_sub = X[:, top_indices].toarray()
        sub_terms = feature_names[top_indices]

        # Rank transform target y
        y_ranked = stats.rankdata(y)

        spearman_r = np.zeros(len(top_indices))
        for i in range(len(top_indices)):
            col = X_sub[:, i]
            if np.std(col) == 0:
                spearman_r[i] = 0.0
            else:
                col_ranked = stats.rankdata(col)
                r, _ = stats.pearsonr(col_ranked, y_ranked)
                spearman_r[i] = 0.0 if np.isnan(r) else float(r)

        df_report = pd.DataFrame({
            "term": sub_terms,
            "spearman_r": spearman_r
        })

        # Sort by absolute spearman_r descending for visualization compatibility
        df_report["abs_r"] = df_report["spearman_r"].abs()
        df_report = df_report.sort_values(by="abs_r", ascending=False).drop(columns=["abs_r"]).reset_index(drop=True)

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "total_terms": n_terms,
            "evaluated_terms": len(sub_terms),
            "max_terms": max_terms,
            "top_terms": df_report.head(10).to_dict(orient="records")
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "max_terms": max_terms},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
