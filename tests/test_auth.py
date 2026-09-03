import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.orchestration import auth
from src.common.settings import settings

app = FastAPI()

@app.get("/x")
def x(role: str = __import__("fastapi").Depends(auth.require("victim"))):
    return {"role": role}

c = TestClient(app)

def test_missing_key():      assert c.get("/x").status_code == 401
def test_bad_key():          assert c.get("/x", headers={"X-API-Key": "nope"}).status_code == 401
def test_wrong_scope():      assert c.get("/x", headers={"X-API-Key": settings.api_keys["counsellor"]}).status_code == 403
def test_ops_override():     assert c.get("/x", headers={"X-API-Key": settings.api_keys["ops"]}).status_code == 200
def test_victim_ok():        assert c.get("/x", headers={"X-API-Key": settings.api_keys["victim"]}).json()["role"] == "victim"