import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session) -> list:
    """Factory function creating DTM tools bound to the session."""

    @tool
    def build_dtm_tool(ngram_range: List[int] = [1, 1]) -> Dict[str, Any]:
        """Build a document-term matrix (DTM) using scikit-learn CountVectorizer.

        Tokenizes text into unigrams or n-grams and generates a word frequency matrix.
        Stores the sparse feature matrix, vocabulary, and fitted CountVectorizer
        in session state.

        Args:
            ngram_range: A list of two integers [min_n, max_n] specifying lower and upper
                bounds for n-grams. Defaults to [1, 1] (unigrams only).

        Returns:
            Dict containing status, result_id, vocabulary_size, n_documents, non_zero,
            total_elements, and sparsity_pct.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No data loaded in session. Call load_dataset_tool first.",
            }

        if "text" not in session.dataframe.columns:
            return {
                "status": "error",
                "message": "Column 'text' missing from session dataframe.",
            }

        # Validate ngram_range parameter
        if (
            not isinstance(ngram_range, (list, tuple))
            or len(ngram_range) != 2
            or not all(isinstance(x, int) and x > 0 for x in ngram_range)
            or ngram_range[0] > ngram_range[1]
        ):
            return {
                "status": "error",
                "message": "ngram_range must be a list of two positive integers with min_n <= max_n.",
            }

        # Initialize vectorizer with default parameters
        vectorizer = CountVectorizer(ngram_range=tuple(ngram_range))

        # Extract text series (handling potential missing values)
        texts = session.dataframe["text"].fillna("")

        # Fit and transform
        X = vectorizer.fit_transform(texts)

        # Update session state
        session.feature_matrix = X
        session.feature_names = list(vectorizer.get_feature_names_out())
        session.artifacts["count_vectorizer"] = vectorizer

        # Set category labels from category_name if available
        if "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
            session.set_labels(labels)

        # Calculate matrix dimensions and sparsity
        n_documents, vocabulary_size = X.shape
        non_zero = int(X.nnz)
        total_elements = int(n_documents * vocabulary_size)
        sparsity_pct = (
            float(100.0 * (1.0 - non_zero / total_elements))
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("dtm")
        summary = {
            "status": "success",
            "result_id": result_id,
            "vocabulary_size": vocabulary_size,
            "n_documents": n_documents,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
        }

        session.store_result(
            tool_name="build_dtm_tool",
            args={"ngram_range": ngram_range},
            summary=summary,
        )

        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in the document-term matrix.

        Sums counts for each term across all documents in session.feature_matrix and
        returns term frequencies sorted in descending order.

        Args:
            top_n: Maximum number of top terms to return in the summary. Defaults to 20.

        Returns:
            Dict containing status, result_id, term_frequencies dict, and top_terms list.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No feature matrix found in session. Call build_dtm_tool first.",
            }

        # Sum counts across columns (axis=0)
        col_sums = np.asarray(session.feature_matrix.sum(axis=0)).ravel()

        # Map terms to total frequencies
        term_freqs = {
            term: int(freq)
            for term, freq in zip(session.feature_names, col_sums)
        }

        # Sort terms by frequency descending
        sorted_terms = sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)
        top_terms = sorted_terms[:top_n]

        result_id = session.next_result_id("term_freq")
        full_report_df = pd.DataFrame(sorted_terms, columns=["term", "frequency"])

        summary = {
            "status": "success",
            "result_id": result_id,
            "total_terms": len(term_freqs),
            "term_frequencies": dict(top_terms),
        }

        session.store_result(
            tool_name="term_frequency_tool",
            args={"top_n": top_n},
            summary=summary,
            full_report=full_report_df,
        )

        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> Dict[str, Any]:
        """Generate a heatmap visualization of a slice of the document-term matrix.

        Slices the global session.feature_matrix up to n_documents (rows) and n_terms (columns),
        renders an annotated heatmap, and sets session.pending_figure.

        Args:
            n_terms: Maximum number of terms (columns) to include from session.feature_names. Defaults to 20.
            n_documents: Maximum number of documents (rows) to include. Defaults to 20.

        Returns:
            Dict containing status, result_id, sliced_matrix, terms, document_indices,
            n_terms_included, and n_documents_included.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No feature matrix found in session. Call build_dtm_tool first.",
            }

        total_docs, total_terms = session.feature_matrix.shape

        # Bound n_documents and n_terms to available matrix dimensions
        actual_n_docs = min(n_documents, total_docs)
        actual_n_terms = min(n_terms, total_terms)

        # Slice matrix slice and convert to dense array
        sub_matrix = session.feature_matrix[:actual_n_docs, :actual_n_terms].toarray()

        terms = session.feature_names[:actual_n_terms]
        doc_indices = list(range(actual_n_docs))

        # Create heatmap figure
        fig, ax = plt.subplots(figsize=(max(6, actual_n_terms * 0.6), max(4, actual_n_docs * 0.4)))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=terms,
            yticklabels=[f"Doc {i}" for i in doc_indices],
            ax=ax,
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Close figure from pyplot state to prevent double rendering in notebooks
        plt.close(fig)

        # Set figure on session for pipeline/notebook display handler
        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "status": "success",
            "result_id": result_id,
            "n_terms_included": actual_n_terms,
            "n_documents_included": actual_n_docs,
            "terms": terms,
            "document_indices": doc_indices,
            "sliced_matrix": sub_matrix.tolist(),
        }

        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
