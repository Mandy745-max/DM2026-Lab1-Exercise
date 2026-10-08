"""Tests for tools_data.py using agent_dev/sample_fixture.csv."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_inspect_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # Test with n=2
    result_n2 = inspect_data_tool.invoke({"n": 2, "max_text_length": 100})
    assert result_n2["status"] == "success"
    assert result_n2["total_rows"] == 8
    assert result_n2["inspected_rows"] == 2

    preview = result_n2["data_preview"]
    assert len(preview) == 2
    assert preview[0]["text"] == "always alpha alpha alpha beta"
    assert preview[0]["category_name"] == "catA"
    assert preview[1]["text"] == "always alpha alpha beta"
    assert preview[1]["category_name"] == "catA"

    # Test with n=20 (larger than dataset size)
    result_n20 = inspect_data_tool.invoke({"n": 20, "max_text_length": 100})
    assert result_n20["status"] == "success"
    assert result_n20["total_rows"] == 8
    assert result_n20["inspected_rows"] == 8
    assert len(result_n20["data_preview"]) == 8


def test_check_missing_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({"text_column": "text"})

    assert result["status"] == "success"
    assert result["total_rows"] == 8
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]


def test_check_duplicates_tool():
    # Test checking duplicates without dropping
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    result = check_duplicates_tool.invoke({"drop": False})

    assert result["status"] == "success"
    assert result["duplicate_count"] == 1
    assert result["duplicate_indices"] == [6]
    assert result["rows_remaining"] == 8

    # Test dropping duplicates (keep=False leaves 6 rows)
    session_drop = SessionState()
    premade_tools_drop, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools_drop if t.name == "load_dataset_tool")

    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools_drop = make_tools(session_drop)
    check_duplicates_tool_drop = next(t for t in data_tools_drop if t.name == "check_duplicates_tool")

    result_drop = check_duplicates_tool_drop.invoke({"drop": True})

    assert result_drop["status"] == "success"
    assert result_drop["duplicate_count"] == 1
    assert result_drop["rows_remaining"] == 6


def test_sample_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert result["status"] == "success"
    assert result["original_rows"] == 8
    assert result["sampled_rows"] == 4
    assert result["sampled_indices"] == [1, 5, 0, 7]
    assert len(session.dataframe) == 4
    assert "original_dataframe" in session.artifacts
    assert len(session.artifacts["original_dataframe"]) == 8


def test_tokenize_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({"text_column": "text"})

    assert result["status"] == "success"
    assert result["total_documents"] == 8
    assert result["total_unigrams"] == 27

    # Check session.dataframe["unigrams"] matches known answer #10
    unigrams = session.dataframe["unigrams"].tolist()
    assert unigrams[0] == ["always", "alpha", "alpha", "alpha", "beta"]
    assert unigrams[1] == ["always", "alpha", "alpha", "beta"]
    assert unigrams[2] == ["always", "alpha", "beta"]
    assert unigrams[3] == ["always", "gamma", "gamma", "gamma", "delta"]
    assert unigrams[4] == ["always", "gamma", "gamma", "delta"]
    assert unigrams[5] == ["always", "gamma", "delta"]
    assert unigrams[6] == ["always", "gamma", "delta"]
    assert unigrams[7] == []


def test_list_files_tool():
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert result["status"] == "success"
    assert result["subdirectory"] == "newdataset"
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_describe_data_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({"text_column": "text"})

    assert result["status"] == "success"

    # Known answer #18 overall stats assertions
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == pytest.approx(0.0)
    assert overall["25%"] == pytest.approx(17.75)
    assert overall["50%"] == pytest.approx(20.5)
    assert overall["75%"] == pytest.approx(25.25)
    assert overall["max"] == pytest.approx(30.0)

    # Category stats assertions
    cat_stats = result["category_stats"]
    assert "catA" in cat_stats
    assert "catB" in cat_stats

    catA = cat_stats["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == pytest.approx(0.0)
    assert catA["25%"] == pytest.approx(12.75)
    assert catA["50%"] == pytest.approx(20.0)
    assert catA["75%"] == pytest.approx(24.5)
    assert catA["max"] == pytest.approx(29.0)

    catB = cat_stats["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == pytest.approx(18.0)
    assert catB["25%"] == pytest.approx(18.0)
    assert catB["50%"] == pytest.approx(21.0)
    assert catB["75%"] == pytest.approx(25.5)
    assert catB["max"] == pytest.approx(30.0)

    # Pending figure check
    assert session.pending_figure is not None
