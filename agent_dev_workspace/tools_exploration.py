"""Exploration tools for corpus analysis, document comparison, and term relationships."""

from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session) -> List[Any]:
    """Factory function returning exploration tools bound to the session."""

    @tool
    def cosine_similarity_tool(doc_index1: int, doc_index2: int) -> Dict[str, Any]:
        """Calculates the cosine similarity between two documents in the corpus.

        Uses the count_vectorizer stored in session.artifacts to vectorize documents
        the same way the DTM did, using raw term counts. If either document index is out
        of range, returns an error status dict. If either document has no terms or zero
        vector norm, returns 0.0 for cosine_similarity.

        Args:
            doc_index1: Integer zero-based index of the first document.
            doc_index2: Integer zero-based index of the second document.

        Returns:
            Dict containing doc_index1, doc_index2, cosine_similarity (float),
            and result_id, or an error status dict if indices are invalid.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "error": "No dataset loaded. Please load a dataset first.",
            }

        df = session.dataframe
        n_docs = len(df)

        if not (0 <= doc_index1 < n_docs and 0 <= doc_index2 < n_docs):
            return {
                "status": "error",
                "error": f"Document index out of bounds. Valid indices are 0 to {n_docs - 1}.",
            }

        # Retrieve count_vectorizer from session.artifacts if present, or vector from feature_matrix
        vectorizer = session.artifacts.get("count_vectorizer")

        if vectorizer is not None and hasattr(vectorizer, "transform"):
            text1 = df.iloc[doc_index1]["text"]
            text2 = df.iloc[doc_index2]["text"]

            # Handle empty/missing text strings directly
            if not isinstance(text1, str) or not text1.strip() or not isinstance(text2, str) or not text2.strip():
                sim = 0.0
            else:
                v1 = vectorizer.transform([text1])
                v2 = vectorizer.transform([text2])

                norm1 = sp.linalg.norm(v1)
                norm2 = sp.linalg.norm(v2)

                if norm1 == 0 or norm2 == 0:
                    sim = 0.0
                else:
                    dot_product = v1.dot(v2.T).toarray()[0, 0]
                    sim = float(dot_product / (norm1 * norm2))
        elif session.feature_matrix is not None:
            v1 = session.feature_matrix[doc_index1]
            v2 = session.feature_matrix[doc_index2]

            norm1 = sp.linalg.norm(v1)
            norm2 = sp.linalg.norm(v2)

            if norm1 == 0 or norm2 == 0:
                sim = 0.0
            else:
                dot_product = v1.dot(v2.T).toarray()[0, 0]
                sim = float(dot_product / (norm1 * norm2))
        else:
            return {
                "status": "error",
                "error": "No fitted count_vectorizer or feature_matrix found in session.",
            }

        if np.isnan(sim):
            sim = 0.0

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "cosine_similarity": float(np.round(sim, 4)),
        }

        session.store_result("cosine_similarity_tool", {"doc_index1": doc_index1, "doc_index2": doc_index2}, summary)
        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> Dict[str, Any]:
        """Computes feature-vs-feature Pearson correlation heatmap for top terms by variance.

        Operates on the global document-term matrix (session.feature_matrix). Selects up to
        top_n terms with highest variance across all documents, computes their Pearson
        correlation matrix, and plots an annotated heatmap stored in session.pending_figure.

        Args:
            top_n: Number of highest-variance terms to include (default: 20).

        Returns:
            Dict containing top_terms (list of str), correlation_matrix (nested list of floats),
            and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "error": "No feature_matrix found in session. Please run build_dtm_tool first.",
            }

        X = session.feature_matrix
        feature_names = np.array(session.feature_names)

        n_samples, n_features = X.shape
        if n_features == 0:
            return {
                "status": "error",
                "error": "Feature matrix has zero terms.",
            }

        # Calculate variance per feature using sparse operations
        if sp.issparse(X):
            mean = np.array(X.mean(axis=0)).ravel()
            mean_sq = np.array(X.power(2).mean(axis=0)).ravel()
            variances = mean_sq - mean**2
        else:
            variances = np.var(X, axis=0)

        # Select top_n features by variance (descending order)
        actual_top_n = min(top_n, n_features)
        top_indices = np.argsort(variances)[::-1][:actual_top_n]

        top_feature_names = feature_names[top_indices].tolist()

        # Slice only the top features and convert only that subset to dense array
        top_submatrix = X[:, top_indices]
        top_dense = top_submatrix.toarray() if sp.issparse(top_submatrix) else np.asarray(top_submatrix)

        # Compute Pearson correlation matrix across top features (columns)
        corr_matrix = np.corrcoef(top_dense.T)

        # If actual_top_n is 1, corrcoef returns a scalar float; handle cleanly
        if actual_top_n == 1:
            corr_matrix = np.array([[1.0]])

        # Handle any NaN correlations (e.g. zero variance columns)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Plot heatmap
        fig, ax = plt.subplots(figsize=(max(6, actual_top_n * 0.4), max(5, actual_top_n * 0.35)))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_feature_names,
            yticklabels=top_feature_names,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Matrix (Top {actual_top_n} Terms by Variance)")
        plt.tight_layout()

        plt.close(fig)
        session.pending_figure = fig

        # Prepare summary result
        result_id = session.next_result_id("feat_corr")
        rounded_corr = np.round(corr_matrix, 4).tolist()

        summary = {
            "result_id": result_id,
            "top_terms": top_feature_names,
            "correlation_matrix": rounded_corr,
        }

        session.store_result("feature_correlation_matrix_tool", {"top_n": top_n}, summary)
        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
