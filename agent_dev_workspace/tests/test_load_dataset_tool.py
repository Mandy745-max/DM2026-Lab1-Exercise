from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool_known_answer_1():
    # 1. Initialize session and load premade tools
    session = SessionState()
    premade_tools, problems = load_workspace_tools("premade_tools", session)
    
    # Check that tools loaded cleanly without error
    assert not problems, f"Failed to load tools from 'premade_tools': {problems}"

    # Find load_dataset_tool
    load_dataset_tool = next(
        (t for t in premade_tools if t.name == "load_dataset_tool"), None
    )
    assert load_dataset_tool is not None, "load_dataset_tool was not found in premade_tools"

    # 2. Invoke load_dataset_tool with the sample fixture
    res = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 3. Assert against Known Answer #1 in TEST_FIXTURE.md
    assert res["n_documents"] == 8
    assert res["counts_per_category"] == {"catA": 4, "catB": 4}
