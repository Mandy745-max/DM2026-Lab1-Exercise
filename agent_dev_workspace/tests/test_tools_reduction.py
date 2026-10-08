"""Tests for tools_reduction.py using sample_fixture.csv."""

import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_known_answers():
    # Set up session and load dataset
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    # Get reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # --- Known Answer #7: PCA ---
    res_pca = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2, "random_state": 42})
    coords_pca = session.artifacts["reduced_coords_pca"]
    
    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    expected_evr = [0.7532, 0.1965]

    assert coords_pca.shape == (8, 2)
    assert np.allclose(coords_pca, expected_pca, atol=1e-3)
    assert np.allclose(res_pca["explained_variance_ratio"], expected_evr, atol=1e-3)
    assert session.pending_figure is not None

    # Reset pending figure check
    session.pending_figure = None

    # --- Known Answer #8: t-SNE ---
    res_tsne = reduce_dimensions_tool.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})
    coords_tsne = session.artifacts["reduced_coords_tsne"]

    assert coords_tsne.shape == (8, 2)
    # Check duplicate rows (rows 5 and 6) receive identical coordinates
    assert np.allclose(coords_tsne[5], coords_tsne[6], atol=1e-3)
    assert session.pending_figure is not None

    # Reset pending figure check
    session.pending_figure = None

    # --- Known Answer #9: UMAP ---
    res_umap = reduce_dimensions_tool.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    coords_umap = session.artifacts["reduced_coords_umap"]

    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])

    assert coords_umap.shape == (8, 2)
    assert np.allclose(coords_umap, expected_umap, atol=1e-1)
    assert session.pending_figure is not None


def test_binarize_labels_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert res["class_names"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]

    matrix = session.artifacts["binarized_labels"]
    assert matrix.shape == (8, 2)

    # Rows 0, 1, 2, 7 should be [1, 0] (catA)
    expected_catA_rows = [0, 1, 2, 7]
    for idx in expected_catA_rows:
        np.testing.assert_array_equal(matrix[idx], [1, 0])

    # Rows 3, 4, 5, 6 should be [0, 1] (catB)
    expected_catB_rows = [3, 4, 5, 6]
    for idx in expected_catB_rows:
        np.testing.assert_array_equal(matrix[idx], [0, 1])
