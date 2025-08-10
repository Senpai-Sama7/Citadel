from fastapi.testclient import TestClient

from services.vector_search.main import app as vector_app
from services.knowledge_graph.main import app as graph_app
from services.causal_inference.main import app as causal_app
from services.time_series.main import app as ts_app
from services.multi_modal.main import app as multi_app
from services.hierarchical_classification.main import app as hier_app
from services.rule_engine.main import app as rule_app
from services.orchestrator.main import app as orch_app
from gateway.main import app as gateway_app


def test_health_endpoints():
    apps = [
        vector_app,
        graph_app,
        causal_app,
        ts_app,
        multi_app,
        hier_app,
        rule_app,
        orch_app,
        gateway_app,
    ]
    for a in apps:
        client = TestClient(a)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"