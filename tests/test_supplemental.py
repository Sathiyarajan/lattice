"""
Supplemental tests targeting branches the main runpy-based integration suite
(test_all_loops.py) doesn't naturally exercise: FastAPI route handlers (only
run on an actual HTTP request), Streamlit UI callback bodies (only run when
a widget has a value), and the "dataset already exists" branch in Loop 40
(only hit on a second run). These raise real line coverage on loop14, 24,
40, 43 closer to their achievable ceiling.
"""
import os
import sys
import runpy

SRC_DIR = os.path.expanduser("~/projects/lattice/src")
sys.path.insert(0, SRC_DIR)


def test_loop14_api_query_endpoint():
    """Exercise the /query FastAPI route handler + cached_retrieve via TestClient."""
    from fastapi.testclient import TestClient
    mod = runpy.run_path(os.path.join(SRC_DIR, "loop14_api.py"), run_name="loop14_api")
    client = TestClient(mod["app"])

    resp1 = client.get("/query", params={"q": "What does Lattice combine?"})
    assert resp1.status_code == 200
    body1 = resp1.json()
    assert "results" in body1 and len(body1["results"]) > 0

    # second identical call should hit the lru_cache path
    resp2 = client.get("/query", params={"q": "What does Lattice combine?"})
    assert resp2.status_code == 200
    assert resp2.json()["results"] == body1["results"]


def test_loop24_streamlit_app_query_flow():
    """Drive the Loop 24 Streamlit app's text_input -> retrieve -> generate branch via AppTest."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(SRC_DIR, "loop24_app.py"), default_timeout=60)
    at.run()
    assert not at.exception

    at.text_input[0].set_value("What does Lattice combine?").run()
    assert not at.exception
    # the app writes at least one markdown/text block once a query is submitted
    assert len(at.markdown) > 0 or len(at.text) > 0


def test_loop43_chatbot_app_chat_flow():
    """Drive the Loop 43 chatbot's chat_input -> session_state -> retrieve/generate branch via AppTest."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(os.path.join(SRC_DIR, "loop43_chatbot_app.py"), default_timeout=60)
    at.run()
    assert not at.exception

    at.chat_input[0].set_value("What does Lattice combine?").run()
    assert not at.exception
    assert len(at.chat_message) >= 2  # user turn + assistant turn rendered


def test_loop40_dataset_already_exists_branch():
    """Running Loop 40 twice hits the 'reuse existing LangSmith dataset' branch instead of creating one."""
    runpy.run_path(os.path.join(SRC_DIR, "loop40_self_corrective_rag.py"), run_name="__main__")
    runpy.run_path(os.path.join(SRC_DIR, "loop40_self_corrective_rag.py"), run_name="__main__")
