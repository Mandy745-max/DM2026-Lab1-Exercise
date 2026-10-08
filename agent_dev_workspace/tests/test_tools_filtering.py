"""Tests for tools_filtering.py (variance_filter_tool, pearson_filter_tool, spearman_filter_tool) matching known answers #5 and #6."""

import pytest
import numpy as np
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    session = SessionState()

    # Load dataset using premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Initialize variance_filter_tool
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run with default / 0.0 threshold to check exact variances
    res = variance_filter_tool.invoke({"threshold": 0.0})

    assert "result_id" in res
    assert res["total_terms"] == 5
    assert res["terms_kept"] == 5
    assert res["terms_removed"] == 0

    # Retrieve full_report stored in results_store
    stored_result = session.results_store[res["result_id"]]
    full_report = stored_result["full_report"]

    # Verify full_report structure (DataFrame, term as 1st col, variance as 2nd col)
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "variance"]

    # Verify descending sort order requirement
    variances_in_report = full_report["variance"].tolist()
    assert variances_in_report == sorted(variances_in_report, reverse=True)

    # Check known population variances from TEST_FIXTURE.md #5
    expected_variances = {
        "alpha": 1.1875,
        "gamma": 1.109375,
        "delta": 0.25,
        "beta": 0.234375,
        "always": 0.109375,
    }

    report_dict = dict(zip(full_report["term"], full_report["variance"]))
    for term, expected_var in expected_variances.items():
        assert report_dict[term] == pytest.approx(expected_var, abs=1e-4)

    # Test threshold filtering (threshold = 0.15)
    res_thresh = variance_filter_tool.invoke({"threshold": 0.15})
    assert res_thresh["terms_kept"] == 4
    assert res_thresh["terms_removed"] == 1

    stored_thresh = session.results_store[res_thresh["result_id"]]
    full_report_thresh = stored_thresh["full_report"]
    kept_terms = full_report_thresh[full_report_thresh["variance"] >= 0.15]["term"].tolist()
    removed_terms = full_report_thresh[full_report_thresh["variance"] < 0.15]["term"].tolist()

    assert set(kept_terms) == {"alpha", "gamma", "delta", "beta"}
    assert set(removed_terms) == {"always"}


def test_pearson_filter_tool_known_answer_6():
    session = SessionState()

    # Load dataset using premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Initialize pearson_filter_tool
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run for target_class="catB"
    res = pearson_filter_tool.invoke({"target_class": "catB"})

    assert "result_id" in res
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5

    # Retrieve full_report stored in results_store
    stored_result = session.results_store[res["result_id"]]
    full_report = stored_result["full_report"]

    # Verify full_report structure (DataFrame, term as 1st col, pearson_r as 2nd col)
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "pearson_r"]

    # Verify sorted by absolute correlation descending
    abs_r_list = full_report["pearson_r"].abs().tolist()
    assert abs_r_list == sorted(abs_r_list, reverse=True)

    # Check known Pearson correlations from TEST_FIXTURE.md #6
    expected_pearson_r = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    report_dict = dict(zip(full_report["term"], full_report["pearson_r"]))
    for term, expected_r in expected_pearson_r.items():
        assert report_dict[term] == pytest.approx(expected_r, abs=1e-3)


def test_spearman_filter_tool_known_answer_6():
    session = SessionState()

    # Load dataset using premade_tools
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using tools_dtm
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Initialize spearman_filter_tool
    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run for target_class="catB"
    res = spearman_filter_tool.invoke({"target_class": "catB"})

    assert "result_id" in res
    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5
    assert res["evaluated_terms"] == 5

    # Retrieve full_report stored in results_store
    stored_result = session.results_store[res["result_id"]]
    full_report = stored_result["full_report"]

    # Verify full_report structure (DataFrame, term as 1st col, spearman_r as 2nd col)
    assert isinstance(full_report, pd.DataFrame)
    assert list(full_report.columns[:2]) == ["term", "spearman_r"]

    # Verify sorted by absolute correlation descending
    abs_r_list = full_report["spearman_r"].abs().tolist()
    assert abs_r_list == sorted(abs_r_list, reverse=True)

    # Check known Spearman correlations from TEST_FIXTURE.md #6
    expected_spearman_r = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    report_dict = dict(zip(full_report["term"], full_report["spearman_r"]))
    for term, expected_r in expected_spearman_r.items():
        assert report_dict[term] == pytest.approx(expected_r, abs=1e-3)
