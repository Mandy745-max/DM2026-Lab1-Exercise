"""Tests for tools_exploration.py tools."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_cosine_similarity_known_answer_16():
    """Test cosine_similarity_tool against known answer #16 in TEST_FIXTURE.md.

    Expected pairs:
    - 0 vs 1: 0.9847
    - 0 vs 3: 0.0909
    - 3 vs 4: 0.9847
    - 0 vs 7: 0.0
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load test fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM to populate session.artifacts["count_vectorizer"] and feature_matrix
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Test doc 0 vs 1
    res_0_1 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 1})
    assert np.isclose(res_0_1["cosine_similarity"], 0.9847, atol=1e-3)

    # Test doc 0 vs 3
    res_0_3 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 3})
    assert np.isclose(res_0_3["cosine_similarity"], 0.0909, atol=1e-3)

    # Test doc 3 vs 4
    res_3_4 = cosine_sim_tool.invoke({"doc_index1": 3, "doc_index2": 4})
    assert np.isclose(res_3_4["cosine_similarity"], 0.9847, atol=1e-3)

    # Test doc 0 vs 7 (empty text)
    res_0_7 = cosine_sim_tool.invoke({"doc_index1": 0, "doc_index2": 7})
    assert res_0_7["cosine_similarity"] == 0.0


def test_feature_correlation_matrix_known_answer_19():
    """Test feature_correlation_matrix_tool against known answer #19 in TEST_FIXTURE.md.

    Expected top terms by variance (descending): ["alpha", "gamma", "delta", "beta", "always"]
    Expected correlation matrix (5x5):
      alpha   gamma   delta    beta  always
    1.0000 -0.5718 -0.6882  0.8885  0.2601
   -0.5718  1.0000  0.8307 -0.6435  0.3140
   -0.6882  0.8307  1.0000 -0.7746  0.3780
    0.8885 -0.6435 -0.7746  1.0000  0.2928
    0.2601  0.3140  0.3780  0.2928  1.0000
    """
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load test fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label",
    })

    # Build DTM to populate session.feature_matrix and session.feature_names
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate exploration tools
    exploration_tools = make_exploration_tools(session)
    feat_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Invoke tool
    res = feat_corr_tool.invoke({"top_n": 20})

    # Assert top terms order by variance
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["top_terms"] == expected_terms

    # Expected correlation matrix from TEST_FIXTURE.md #19
    expected_corr = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]

    actual_corr = np.array(res["correlation_matrix"])
    assert np.allclose(actual_corr, expected_corr, atol=1e-3)

    # Assert that pending_figure is set
    assert session.pending_figure is not None
