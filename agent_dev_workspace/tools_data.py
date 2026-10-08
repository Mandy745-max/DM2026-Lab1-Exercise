"""Tools for data loading, inspection, missing value check, duplication check, sampling, tokenization, and descriptive stats."""

import os
from typing import Dict, Any, Optional
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    """Factory function creating data tools attached to the provided session state."""

    @tool
    def inspect_data_tool(n: int = 5, max_text_length: int = 100) -> Dict[str, Any]:
        """Peek at the first n rows of the working dataset with truncated text and category columns.

        Args:
            n: Number of rows to inspect from the top of session.dataframe. Defaults to 5.
            max_text_length: Maximum number of characters to display for text fields. Defaults to 100.

        Returns:
            Dict containing result_id, status, total_rows, inspected_rows, and data_preview.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded. Please load a dataset first."
            }

        total_rows = len(session.dataframe)
        # Cap n to total_rows if larger than dataset size
        rows_to_show = min(n, total_rows)
        slice_df = session.dataframe.head(rows_to_show)

        cols_to_include = [c for c in ["text", "category_name", "category"] if c in slice_df.columns]

        preview = []
        for idx, row in slice_df.iterrows():
            row_dict = {"index": int(idx)}
            for col in cols_to_include:
                val = row[col]
                if col == "text" and isinstance(val, str) and len(val) > max_text_length:
                    row_dict[col] = val[:max_text_length] + "..."
                else:
                    row_dict[col] = val
            preview.append(row_dict)

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "inspected_rows": len(preview),
            "data_preview": preview,
        }

        session.store_result("inspect_data_tool", {"n": n, "max_text_length": max_text_length}, summary)
        return summary

    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """Check for missing or empty text values in the loaded dataset.

        Args:
            text_column: The column name in session.dataframe to check for missing or empty text.
                         Defaults to "text".

        Returns:
            Dict containing result_id, missing_count, missing_indices, total_rows, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded. Please load a dataset first."
            }

        if text_column not in session.dataframe.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in dataset."
            }

        col = session.dataframe[text_column]

        # Identify missing (NaN/None) or empty/whitespace-only strings
        is_null = col.isna()
        is_empty = col.dropna().astype(str).str.strip() == ""

        # Combine null and empty masks
        missing_mask = is_null.copy()
        missing_mask.loc[is_empty.index] |= is_empty

        missing_indices = session.dataframe.index[missing_mask].tolist()
        missing_count = len(missing_indices)
        total_rows = len(session.dataframe)

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": total_rows,
            "missing_count": missing_count,
            "missing_indices": missing_indices,
            "text_column": text_column,
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> Dict[str, Any]:
        """Check for duplicate documents in the dataset and optionally drop them.

        Args:
            drop: If True, drops all copies of duplicated rows using keep=False. Defaults to False.

        Returns:
            Dict containing result_id, duplicate_count, duplicate_indices, rows_remaining, dropped, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded. Please load a dataset first."
            }

        # Compare on explicit column subset to avoid unhashable types (e.g. unigrams list)
        subset_cols = [c for c in ["text", "category", "category_name"] if c in session.dataframe.columns]

        # Count duplicates with keep="first" so duplicate_count = extra copies found (e.g. 1)
        dup_mask = session.dataframe.duplicated(subset=subset_cols, keep="first")
        dup_indices = session.dataframe.index[dup_mask].tolist()
        dup_count = len(dup_indices)

        if drop and dup_count > 0:
            # Drop using keep=False across specified subset (matching Master's drop behavior)
            session.dataframe.drop_duplicates(subset=subset_cols, keep=False, inplace=True)
            session.dataframe.reset_index(drop=True, inplace=True)

        rows_remaining = len(session.dataframe)
        result_id = session.next_result_id("check_duplicates")

        summary = {
            "result_id": result_id,
            "status": "success",
            "duplicate_count": dup_count,
            "duplicate_indices": dup_indices,
            "rows_remaining": rows_remaining,
            "dropped": drop,
        }

        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: Optional[int] = None) -> Dict[str, Any]:
        """Subsample the working dataset (session.dataframe) to n rows.

        Args:
            n: Number of rows to sample.
            random_state: Optional seed for reproducible sampling.

        Returns:
            Dict containing result_id, original_rows, sampled_rows, sampled_indices, random_state, and status.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded. Please load a dataset first."
            }

        original_rows = len(session.dataframe)

        if n > original_rows:
            return {
                "status": "error",
                "message": f"Requested sample size n={n} is greater than dataset size {original_rows}."
            }

        # Store original dataframe in artifacts before replacing
        if "original_dataframe" not in session.artifacts:
            session.artifacts["original_dataframe"] = session.dataframe.copy()

        sampled_df = session.dataframe.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        session.dataframe = sampled_df.reset_index(drop=True)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "original_rows": original_rows,
            "sampled_rows": len(session.dataframe),
            "sampled_indices": sampled_indices,
            "random_state": random_state,
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def tokenize_tool(text_column: str = "text") -> Dict[str, Any]:
        """Tokenize each document's text into unigrams (lowercase word tokens) using nltk.word_tokenize and store in session.dataframe['unigrams'].

        Args:
            text_column: Name of the column in session.dataframe to tokenize. Defaults to "text".

        Returns:
            Dict containing result_id, status, total_documents, total_unigrams, and text_column.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded. Please load a dataset first."
            }

        if text_column not in session.dataframe.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in dataset."
            }

        # Tokenize using nltk.word_tokenize on lowercased text
        def tokenize_text(text):
            if not isinstance(text, str) or not text.strip():
                return []
            return nltk.word_tokenize(text.lower())

        session.dataframe["unigrams"] = session.dataframe[text_column].apply(tokenize_text)
        total_unigrams = int(session.dataframe["unigrams"].apply(len).sum())
        total_documents = len(session.dataframe)

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_documents": total_documents,
            "total_unigrams": total_unigrams,
            "text_column": text_column,
        }

        session.store_result("tokenize_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """List files and folders in a specified directory within the repository.

        Args:
            subdirectory: Path relative to repository root to list files from. Defaults to ".".

        Returns:
            Dict containing result_id, status, subdirectory, files, and directories.
        """
        repo_root = os.path.abspath(os.getcwd())
        target_path = os.path.abspath(os.path.join(repo_root, subdirectory))

        # Security check: ensure target path is within repository root
        if not target_path.startswith(repo_root):
            return {
                "status": "error",
                "message": f"Access denied: '{subdirectory}' is outside repository root."
            }

        if not os.path.exists(target_path):
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' does not exist."
            }

        if not os.path.isdir(target_path):
            return {
                "status": "error",
                "message": f"Path '{subdirectory}' is not a directory."
            }

        entries = sorted(os.listdir(target_path))
        files = [e for e in entries if os.path.isfile(os.path.join(target_path, e))]
        directories = [e for e in entries if os.path.isdir(os.path.join(target_path, e))]

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "files": files,
            "directories": directories,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def describe_data_tool(text_column: str = "text") -> Dict[str, Any]:
        """Compute summary statistics on document text length and generate a box plot by category.

        Args:
            text_column: Name of the column in session.dataframe containing document text. Defaults to "text".

        Returns:
            Dict containing result_id, status, overall_stats, category_stats, and text_column.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset loaded. Please load a dataset first."
            }

        if text_column not in session.dataframe.columns:
            return {
                "status": "error",
                "message": f"Column '{text_column}' not found in dataset."
            }

        df = session.dataframe.copy()
        df["text_length"] = df[text_column].astype(str).apply(len)

        overall_stats = df["text_length"].describe().to_dict()

        category_stats = {}
        if "category_name" in df.columns:
            cat_desc = df.groupby("category_name")["text_length"].describe()
            category_stats = {cat: row.to_dict() for cat, row in cat_desc.iterrows()}

        fig, ax = plt.subplots(figsize=(6, 4))
        if "category_name" in df.columns and df["category_name"].nunique() > 0:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")
        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        plt.close(fig)
        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "overall_stats": overall_stats,
            "category_stats": category_stats,
            "text_column": text_column,
        }

        session.store_result("describe_data_tool", {"text_column": text_column}, summary)
        return summary

    return [
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        tokenize_tool,
        list_files_tool,
        describe_data_tool,
    ]
