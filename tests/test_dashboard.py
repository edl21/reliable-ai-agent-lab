"""Optional dashboard route tests."""

from fastapi.testclient import TestClient

from app.main import app


def test_dashboard_page_and_data_are_available() -> None:
    client = TestClient(app)
    page = client.get("/dashboard")
    data = client.get("/dashboard/data")
    assert page.status_code == 200
    assert "Reliable AI Agent Lab" in page.text
    assert data.status_code == 200
    assert isinstance(data.json()["chapters"], list)
    assert isinstance(data.json()["traces"], list)


def test_dashboard_rejects_unknown_suite() -> None:
    response = TestClient(app).post("/dashboard/run/not-allowlisted")
    assert response.status_code == 404
