from fastapi.testclient import TestClient

from gateway.main import app as gateway_app


def test_list_services(monkeypatch):
    # Provide dummy URLs so the gateway can resolve them
    monkeypatch.setenv("VECTOR_SEARCH_URL", "http://vector_search:8000")
    monkeypatch.setenv("KNOWLEDGE_GRAPH_URL", "http://knowledge_graph:8000")
    monkeypatch.setenv("CAUSAL_INFERENCE_URL", "http://causal_inference:8000")
    monkeypatch.setenv("TIME_SERIES_URL", "http://time_series:8000")
    monkeypatch.setenv("MULTI_MODAL_URL", "http://multi_modal:8000")
    monkeypatch.setenv(
        "HIERARCHICAL_CLASSIFICATION_URL", "http://hierarchical_classification:8000"
    )
    monkeypatch.setenv("RULE_ENGINE_URL", "http://rule_engine:8000")
    monkeypatch.setenv("ORCHESTRATOR_URL", "http://orchestrator:8000")
    client = TestClient(gateway_app)
    resp = client.get("/services")
    assert resp.status_code == 200
    body = resp.json()
    # Ensure that each service prefix is present in the response
    for prefix in [
        "vector",
        "knowledge",
        "causal",
        "time",
        "multi",
        "hier",
        "rule",
        "orch",
    ]:
        assert prefix in body