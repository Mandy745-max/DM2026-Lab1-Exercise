"""Tests for tools_patterns.py (mine_patterns_tool) using agent_dev/sample_fixture.csv."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def setup_fixture_session():
    """Helper to initialize state and load sample_fixture.csv dataset."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")
    return session, mine_patterns_tool


def test_mine_patterns_variance_and_tf_zero_terms():
    """Test known answer #17: variance and term_frequency keep zero terms for both categories."""
    _, mine_patterns_tool = setup_fixture_session()

    for cat in ["catA", "catB"]:
        for method in ["variance", "term_frequency"]:
            res = mine_patterns_tool.invoke({
                "category_name": cat,
                "filtering_method": method,
                "algorithm": "fpgrowth",
                "min_sup": 1
            })
            assert res["status"] == "success"
            assert res["patterns"] == []
            assert res["n_patterns"] == 0
            assert res.get("note") == "no terms survived filtering"


def test_mine_patterns_tfidf_fpgrowth():
    """Test known answer #17: tfidf filtering with fpgrowth algorithm."""
    _, mine_patterns_tool = setup_fixture_session()

    # catA at min_sup=3 (should find {alpha} support 3)
    res_a3 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    assert res_a3["status"] == "success"
    assert res_a3["n_patterns"] == 1
    assert res_a3["patterns"][0]["items"] == ["alpha"]
    assert res_a3["patterns"][0]["support"] == 3

    # catA at min_sup=4 (should find 0 patterns)
    res_a4 = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_a4["status"] == "success"
    assert res_a4["n_patterns"] == 0
    assert res_a4["patterns"] == []

    # catB at min_sup=4 (should find {gamma} support 4)
    res_b4 = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_b4["status"] == "success"
    assert res_b4["n_patterns"] == 1
    assert res_b4["patterns"][0]["items"] == ["gamma"]
    assert res_b4["patterns"][0]["support"] == 4


def test_mine_patterns_topk():
    """Test known answer #20: topk algorithm (FAE)."""
    _, mine_patterns_tool = setup_fixture_session()

    # catA with topk k=1
    res_a_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_a_topk["status"] == "success"
    assert res_a_topk["n_patterns"] == 1
    assert res_a_topk["patterns"][0]["items"] == ["alpha"]
    assert res_a_topk["patterns"][0]["support"] == 3

    # catB with topk k=1
    res_b_topk = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_b_topk["status"] == "success"
    assert res_b_topk["n_patterns"] == 1
    assert res_b_topk["patterns"][0]["items"] == ["gamma"]
    assert res_b_topk["patterns"][0]["support"] == 4


def test_mine_patterns_maxfpgrowth():
    """Test known answer #20: maxfpgrowth algorithm returns zero patterns on single-term dataset."""
    _, mine_patterns_tool = setup_fixture_session()

    for cat in ["catA", "catB"]:
        res_max = mine_patterns_tool.invoke({
            "category_name": cat,
            "filtering_method": "tfidf",
            "algorithm": "maxfpgrowth",
            "min_sup": 1
        })
        assert res_max["status"] == "success"
        assert res_max["n_patterns"] == 0
        assert res_max["patterns"] == []
