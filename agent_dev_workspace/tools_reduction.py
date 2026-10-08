"""Dimensionality reduction and label processing tools for data mining pipeline."""

from typing import Dict, Any, Optional, Literal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from langchain_core.tools import tool

from agent_pipeline.session_state import SessionState


def make_tools(session: SessionState):
    """Factory function that returns tools for reduction and label processing.

    Args:
        session: Shared SessionState pipeline instance.

    Returns:
        List of initialized tools.
    """

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Reduce feature matrix dimensions using PCA, t-SNE, or UMAP, and plot the result.

        Performs dimensionality reduction on session.feature_matrix and generates
        a scatter plot colored by document category.

        Args:
            method: Reduction algorithm ('pca', 'tsne', or 'umap'). Default 'pca'.
            n_components: Number of target dimensions (typically 2). Default 2.
            perplexity: Perplexity parameter for t-SNE. Default 30.0.
            n_neighbors: Number of neighbors parameter for UMAP. Default 15.
            random_state: Random state seed for reproducibility. Default 42.

        Returns:
            Dictionary containing result_id, method, coordinates summary (first 5 samples),
            and additional metadata such as explained_variance_ratio for PCA.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found in session. Build DTM first."}

        X = session.feature_matrix
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X

        method_lower = method.lower()
        explained_variance = None

        try:
            if method_lower == "pca":
                reducer = PCA(n_components=n_components, random_state=random_state)
                coords = reducer.fit_transform(X_dense)
                explained_variance = reducer.explained_variance_ratio_.tolist()
            elif method_lower == "tsne":
                reducer = TSNE(
                    n_components=n_components,
                    perplexity=perplexity,
                    random_state=random_state,
                )
                coords = reducer.fit_transform(X_dense)
            elif method_lower == "umap":
                reducer = umap.UMAP(
                    n_components=n_components,
                    n_neighbors=n_neighbors,
                    random_state=random_state,
                )
                coords = reducer.fit_transform(X_dense)
            else:
                return {"error": f"Unsupported method: {method}. Use 'pca', 'tsne', or 'umap'."}
        except Exception as e:
            return {"error": f"Failed to perform {method_lower} reduction: {str(e)}"}

        # Store full reduced coordinates in session artifacts
        artifact_key = f"reduced_coords_{method_lower}"
        session.artifacts[artifact_key] = coords

        # Generate scatter plot colored by category
        fig, ax = plt.subplots(figsize=(8, 6))

        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            categories = session.dataframe["category_name"].values
        elif session.dataframe is not None and "category" in session.dataframe.columns:
            categories = session.dataframe["category"].values
        elif session.labels is not None:
            categories = session.labels
        else:
            categories = np.array(["all"] * coords.shape[0])

        unique_cats = np.unique(categories)
        for cat in unique_cats:
            mask = categories == cat
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                label=str(cat),
                alpha=0.7,
                edgecolors="none",
            )

        ax.set_title(f"2D Projection ({method_lower.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(title="Category")
        plt.tight_layout()

        # Close figure before assigning to session.pending_figure as requested
        plt.close(fig)
        session.pending_figure = fig

        res_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": res_id,
            "method": method_lower,
            "n_components": n_components,
            "n_samples": coords.shape[0],
            "coordinates": coords[:5].tolist(),
        }
        if explained_variance is not None:
            summary["explained_variance_ratio"] = explained_variance

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary=summary,
        )

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """Binarize category labels into a one-hot encoded matrix.

        One-hot encodes category_name in session.dataframe into binary indicator columns,
        one column per class sorted alphabetically.

        Returns:
            Dictionary containing result_id, class_names, matrix shape, and preview of first 5 rows.
        """
        if session.dataframe is None:
            return {"error": "No DataFrame found in session. Load dataset first."}

        df = session.dataframe
        if "category_name" in df.columns:
            labels_series = df["category_name"]
        elif "category" in df.columns:
            labels_series = df["category"]
        elif session.labels is not None:
            labels_series = pd.Series(session.labels)
        else:
            return {"error": "No category labels found in session.dataframe or session.labels."}

        # One-hot encode labels ensuring 1 column per category sorted by name
        binarized_df = pd.get_dummies(labels_series, dtype=int)
        # Sort column names explicitly
        sorted_cols = sorted(binarized_df.columns.tolist())
        binarized_df = binarized_df[sorted_cols]

        binarized_matrix = binarized_df.values
        class_names = [str(c) for c in sorted_cols]

        session.artifacts["binarized_labels"] = binarized_matrix
        session.artifacts["binarized_label_names"] = class_names

        res_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": res_id,
            "class_names": class_names,
            "shape": list(binarized_matrix.shape),
            "preview_first_5_rows": binarized_matrix[:5].tolist(),
        }

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
