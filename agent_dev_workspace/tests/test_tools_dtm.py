import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_known_answer_4():
    """Test build_dtm_tool against known answer #4 in TEST_FIXTURE.md."""
    # Create single shared session state
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )

    # Load sample fixture CSV
    load_res = load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    assert load_res["n_documents"] == 8

    # Load student's build_dtm_tool via direct import
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")

    # Execute build_dtm_tool
    result = build_dtm_tool.invoke({"ngram_range": [1, 1]})

    # Assert return summary metrics
    assert result["status"] == "success"
    assert result["n_documents"] == 8
    assert result["vocabulary_size"] == 5
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert pytest.approx(result["sparsity_pct"], abs=1e-3) == 47.5

    # Assert session state attributes
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_known_answer_14():
    """Test term_frequency_tool against known answer #14 in TEST_FIXTURE.md."""
    # Create fresh session state for isolated execution
    session = SessionState()

    # Load dataset using premade tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Initialize tools_dtm tools
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in tools if t.name == "term_frequency_tool")

    # Build DTM first to populate session.feature_matrix and session.feature_names
    build_dtm_tool.invoke({"ngram_range": [1, 1]})

    # Execute term_frequency_tool passing top_n=10
    result = term_frequency_tool.invoke({"top_n": 10})

    assert result["status"] == "success"
    assert result["total_terms"] == 5

    # Assert term_frequencies dict against known answer #14
    term_freqs = result["term_frequencies"]
    assert term_freqs["alpha"] == 6
    assert term_freqs["always"] == 7
    assert term_freqs["beta"] == 3
    assert term_freqs["delta"] == 4
    assert term_freqs["gamma"] == 7


def test_dtm_heatmap_known_answer_21():
    """Test dtm_heatmap_tool against known answer #21 in TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset using premade tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(
        t for t in premade_tools if t.name == "load_dataset_tool"
    )
    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    # Initialize tools_dtm tools
    tools = make_tools(session)
    build_dtm_tool = next(t for t in tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in tools if t.name == "dtm_heatmap_tool")

    # Build DTM
    build_dtm_tool.invoke({"ngram_range": [1, 1]})

    # Test 1: Sliced call with n_terms=2, n_documents=3
    sliced_res = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert sliced_res["status"] == "success"
    assert sliced_res["n_terms_included"] == 2
    assert sliced_res["n_documents_included"] == 3
    assert sliced_res["terms"] == ["alpha", "always"]
    assert sliced_res["sliced_matrix"] == [[3, 1], [2, 1], [1, 1]]

    # Test 2: Default parameters (returns full 8x5 matrix for sample_fixture.csv)
    default_res = dtm_heatmap_tool.invoke({})
    assert default_res["status"] == "success"
    assert default_res["n_terms_included"] == 5
    assert default_res["n_documents_included"] == 8
    assert default_res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]
    assert default_res["sliced_matrix"] == expected_full_matrix

    # Verify that session.pending_figure is set
    assert session.pending_figure is not None
