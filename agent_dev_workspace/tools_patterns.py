"""Tools for frequent-pattern mining on text documents."""

import os
from typing import Dict, Any, List, Optional
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

# Import PAMI pattern mining modules
import PAMI.extras.convert.DF2DB as db_converter
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session) -> list:
    """Creates pattern mining tools bound to the given session state."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "variance",
        algorithm: str = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mines frequent itemsets/patterns from document text for a specific category.

        Performs per-category local vectorization, applies word vocabulary filtering
        (variance, tfidf, or term_frequency), converts transaction data, and runs
        PAMI algorithms (fpgrowth, topk, or maxfpgrowth).

        Args:
            category_name: The target category to filter documents by.
            filtering_method: Method to filter vocabulary ('variance', 'tfidf', or 'term_frequency').
            algorithm: Pattern mining algorithm to run ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support threshold (integer count), required for 'fpgrowth' and 'maxfpgrowth'.
            k: Top-k count, required for 'topk'.

        Returns:
            Dict with status, category, algorithm, filtering_method, pattern counts, and mined patterns.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session."}

        # Filter dataframe by category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {
                "status": "error",
                "message": f"No documents found for category '{category_name}'."
            }

        # Step 1: Category-local CountVectorizer with English stop words removed
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm = vectorizer.fit_transform(cat_df["text"])
        except ValueError:
            # e.g., if text consists solely of stop words or empty strings
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived vectorization/stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived vectorization/stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 2: Vocabulary filtering based on percentile thresholds
        dense_dtm = dtm.toarray()
        kept_indices = []

        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                scores = dense_dtm.var(axis=0)
            else:
                scores = dense_dtm.sum(axis=0)

            p5 = pd.Series(scores).quantile(0.05)
            p95 = pd.Series(scores).quantile(0.95)
            kept_indices = [i for i, score in enumerate(scores) if p5 < score < p95]

        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm).toarray()
            scores = tfidf_mat.mean(axis=0)
            p20 = pd.Series(scores).quantile(0.20)
            kept_indices = [i for i, score in enumerate(scores) if score >= p20]
        else:
            return {
                "status": "error",
                "message": f"Unsupported filtering_method: '{filtering_method}'"
            }

        # Step 3: Handle zero terms surviving filtering cleanly
        if not kept_indices:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "status": "success",
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "n_patterns": 0,
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        surviving_features = [feature_names[i] for i in kept_indices]
        surviving_dtm = dense_dtm[:, kept_indices]

        # Step 4: Convert surviving terms to transactional database format
        binary_df = pd.DataFrame(surviving_dtm, columns=surviving_features)

        db_path = "temp_pami_db.txt"

        # DF2DB and convert2TransactionalDatabase matching Master notebook signature
        converter = db_converter.DF2DB(binary_df)
        converter.convert2TransactionalDatabase(db_path, ">=", 1)

        # Handle encoding gracefully: decode cp1252 if UTF-8 fails, then save back as UTF-8
        if os.path.exists(db_path):
            try:
                with open(db_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(db_path, "r", encoding="cp1252") as f:
                    content = f.read()
            with open(db_path, "w", encoding="utf-8") as f:
                f.write(content)

        # Execute selected mining algorithm using .mine()
        try:
            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"status": "error", "message": "min_sup is required for fpgrowth"}
                miner = FPGrowth.FPGrowth(db_path, minSup=min_sup)
                miner.mine()
                patterns_dict = miner.getPatterns()
            elif algorithm == "topk":
                if k is None:
                    return {"status": "error", "message": "k is required for topk algorithm"}
                miner = FAE.FAE(db_path, k=k)
                miner.mine()
                patterns_dict = miner.getPatterns()
            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"status": "error", "message": "min_sup is required for maxfpgrowth"}
                miner = MaxFPGrowth.MaxFPGrowth(db_path, minSup=min_sup)
                miner.mine()
                patterns_dict = miner.getPatterns()
            else:
                return {"status": "error", "message": f"Unsupported algorithm: '{algorithm}'"}
        except Exception as e:
            return {"status": "error", "message": f"Mining error: {str(e)}"}
        finally:
            # Clean up temporary database file after mining
            if os.path.exists(db_path):
                os.remove(db_path)

        # Format output patterns list, handling keys as tuples, lists, or space-separated strings
        all_patterns = []
        if patterns_dict:
            for itemset, support in patterns_dict.items():
                if isinstance(itemset, (tuple, list)):
                    items = [str(x) for x in itemset]
                elif isinstance(itemset, str):
                    items = itemset.strip().split()
                else:
                    items = [str(itemset)]
                all_patterns.append({"items": items, "support": int(support)})

        # Sort patterns descending by support
        all_patterns.sort(key=lambda x: x["support"], reverse=True)

        # Keep top 20 patterns for summary output
        top_patterns = all_patterns[:20]

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "status": "success",
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(all_patterns),
            "patterns": top_patterns
        }

        # Store full pattern list in session artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = all_patterns
        session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)

        return summary

    return [mine_patterns_tool]
