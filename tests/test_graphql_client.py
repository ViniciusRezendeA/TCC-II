from __future__ import annotations

import datetime
from unittest.mock import MagicMock

import pytest

from mcp_pipeline.github.graphql_client import (
    GraphQLClient,
    GraphQLError,
    RetryableHTTPError,
)


def make_response(status_code=200, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    resp.json.return_value = json_data or {}
    if status_code >= 400 and status_code not in (403,):
        resp.raise_for_status.side_effect = Exception(f"HTTP {status_code}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def future_iso(seconds: float) -> str:
    dt = datetime.datetime.now(datetime.UTC) + datetime.timedelta(seconds=seconds)
    return dt.isoformat().replace("+00:00", "Z")


def test_execute_returns_data_on_success(monkeypatch):
    client = GraphQLClient(token="fake-token")
    ok_response = make_response(
        200,
        {
            "data": {
                "rateLimit": {"cost": 1, "remaining": 4999, "resetAt": future_iso(3600)},
                "viewer": {"login": "octocat"},
            }
        },
    )
    monkeypatch.setattr(client._session, "post", MagicMock(return_value=ok_response))

    data = client.execute("query { viewer { login } rateLimit { cost remaining resetAt } }")

    assert data["viewer"]["login"] == "octocat"
    assert client.last_rate_limit["remaining"] == 4999


def test_execute_raises_graphql_error_on_errors_payload(monkeypatch):
    client = GraphQLClient(token="fake-token")
    error_response = make_response(200, {"errors": [{"message": "Something went wrong"}]})
    monkeypatch.setattr(client._session, "post", MagicMock(return_value=error_response))

    with pytest.raises(GraphQLError):
        client.execute("query { viewer { login } }")


def test_execute_retries_on_5xx_then_succeeds(monkeypatch):
    client = GraphQLClient(token="fake-token")
    monkeypatch.setattr("mcp_pipeline.github.graphql_client.time.sleep", lambda *_: None)

    responses = [
        make_response(503),
        make_response(
            200,
            {"data": {"rateLimit": {"cost": 1, "remaining": 100, "resetAt": future_iso(3600)}}},
        ),
    ]
    monkeypatch.setattr(client._session, "post", MagicMock(side_effect=responses))

    data = client.execute("query { rateLimit { cost remaining resetAt } }")
    assert data["rateLimit"]["remaining"] == 100


def test_execute_gives_up_after_max_attempts(monkeypatch):
    client = GraphQLClient(token="fake-token")
    monkeypatch.setattr("mcp_pipeline.github.graphql_client.time.sleep", lambda *_: None)
    monkeypatch.setattr(client._session, "post", MagicMock(return_value=make_response(503)))

    with pytest.raises(RetryableHTTPError):
        client.execute("query { rateLimit { cost remaining resetAt } }")


def test_pause_if_budget_low_sleeps_when_remaining_below_threshold(monkeypatch):
    client = GraphQLClient(token="fake-token", min_remaining_before_pause=50)
    sleep_calls = []
    monkeypatch.setattr(
        "mcp_pipeline.github.graphql_client.time.sleep", lambda s: sleep_calls.append(s)
    )
    low_budget_response = make_response(
        200,
        {"data": {"rateLimit": {"cost": 1, "remaining": 10, "resetAt": future_iso(120)}}},
    )
    monkeypatch.setattr(client._session, "post", MagicMock(return_value=low_budget_response))

    client.execute("query { rateLimit { cost remaining resetAt } }")

    assert len(sleep_calls) == 1
    assert sleep_calls[0] > 0


def test_does_not_sleep_when_budget_healthy(monkeypatch):
    client = GraphQLClient(token="fake-token", min_remaining_before_pause=50)
    sleep_calls = []
    monkeypatch.setattr(
        "mcp_pipeline.github.graphql_client.time.sleep", lambda s: sleep_calls.append(s)
    )
    healthy_response = make_response(
        200,
        {"data": {"rateLimit": {"cost": 1, "remaining": 4000, "resetAt": future_iso(3600)}}},
    )
    monkeypatch.setattr(client._session, "post", MagicMock(return_value=healthy_response))

    client.execute("query { rateLimit { cost remaining resetAt } }")

    assert sleep_calls == []
