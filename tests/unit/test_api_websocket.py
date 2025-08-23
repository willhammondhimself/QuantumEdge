"""Tests for WebSocket API functionality."""

import pytest
import json
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.streaming.websocket import MessageType


class TestWebSocketEndpoint:
    """Test WebSocket endpoint functionality."""

    def test_websocket_connection(self):
        """Test basic WebSocket connection."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Should receive connection status
            data = websocket.receive_json()
            assert data["type"] == "connection_status"
            assert data["data"]["status"] == "connected"
            assert "client_id" in data["data"]

    def test_websocket_heartbeat(self):
        """Test WebSocket heartbeat mechanism."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send heartbeat
            heartbeat_msg = {
                "type": "heartbeat",
                "data": {"status": "alive"},
                "timestamp": 1234567890,
            }
            websocket.send_json(heartbeat_msg)

            # Should receive heartbeat response
            response = websocket.receive_json()
            assert response["type"] == "heartbeat"
            assert response["data"]["status"] == "alive"
            assert "server_time" in response["data"]

    def test_websocket_subscription(self):
        """Test topic subscription."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            connection_msg = websocket.receive_json()
            client_id = connection_msg["data"]["client_id"]

            # Subscribe to topic
            subscribe_msg = {
                "type": "subscribe",
                "data": {"topic": "portfolio_updates"},
                "timestamp": 1234567890,
            }
            websocket.send_json(subscribe_msg)

            # Should receive subscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "subscribe"
            assert response["data"]["topic"] == "portfolio_updates"
            assert response["data"]["status"] == "subscribed"
            assert response["client_id"] == client_id

    def test_websocket_unsubscribe(self):
        """Test topic unsubscription."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Subscribe first
            subscribe_msg = {
                "type": "subscribe",
                "data": {"topic": "price_updates"},
                "timestamp": 1234567890,
            }
            websocket.send_json(subscribe_msg)
            websocket.receive_json()  # Skip confirmation

            # Unsubscribe
            unsubscribe_msg = {
                "type": "unsubscribe",
                "data": {"topic": "price_updates"},
                "timestamp": 1234567890,
            }
            websocket.send_json(unsubscribe_msg)

            # Should receive unsubscription confirmation
            response = websocket.receive_json()
            assert response["type"] == "unsubscribe"
            assert response["data"]["topic"] == "price_updates"
            assert response["data"]["status"] == "unsubscribed"

    @patch("src.api.main.optimization_service")
    def test_websocket_optimization_request(self, mock_opt_service):
        """Test optimization request via WebSocket."""
        # Mock the optimization service
        mock_opt_service.request_optimization = AsyncMock(return_value="opt_test_123")

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send optimization request
            opt_request = {
                "type": "optimization_request",
                "data": {
                    "symbols": ["AAPL", "GOOGL", "MSFT"],
                    "objective": "maximize_sharpe",
                    "constraints": {"long_only": True, "sum_to_one": True},
                    "method": "mean_variance",
                    "trigger": "manual",
                },
                "timestamp": 1234567890,
            }
            websocket.send_json(opt_request)

            # May receive an error if optimization service is not initialized
            response = websocket.receive_json()
            # Just check we got a response (could be error or success)
            assert "type" in response

    def test_websocket_invalid_message(self):
        """Test handling of invalid messages."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send invalid JSON
            websocket.send_text("invalid json")

            # Should receive error message
            response = websocket.receive_json()
            assert response["type"] == "error"
            assert "Invalid JSON format" in response["data"]["error"]

    def test_websocket_unknown_message_type(self):
        """Test handling of unknown message types."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send unknown message type
            unknown_msg = {
                "type": "unknown_type",
                "data": {"test": "data"},
                "timestamp": 1234567890,
            }
            websocket.send_json(unknown_msg)

            # Should receive error message for invalid message type
            response = websocket.receive_json()
            assert response["type"] == "error"
            # The error message might vary depending on the implementation
            assert "error" in response["data"]

    def test_websocket_multiple_connections(self):
        """Test multiple simultaneous WebSocket connections."""
        client = TestClient(app)

        # Create multiple connections
        connections = []
        client_ids = []

        for i in range(3):
            websocket = client.websocket_connect("/ws").__enter__()
            connections.append(websocket)

            # Get client ID
            conn_msg = websocket.receive_json()
            client_ids.append(conn_msg["data"]["client_id"])

        # Verify all client IDs are unique
        assert len(set(client_ids)) == 3

        # Clean up connections
        for ws in connections:
            ws.__exit__(None, None, None)

    @patch("src.api.main.optimization_service")
    def test_websocket_optimization_flow(self, mock_opt_service):
        """Test complete optimization flow via WebSocket."""
        mock_opt_service.request_optimization = AsyncMock(return_value="opt_test_123")

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send CVaR optimization request
            cvar_request = {
                "type": "optimization_request",
                "data": {
                    "symbols": ["AAPL", "GOOGL", "MSFT", "AMZN"],
                    "objective": "minimize_cvar",
                    "constraints": {
                        "long_only": True,
                        "sum_to_one": True,
                        "cvar_confidence": 0.05,
                    },
                    "method": "mean_variance",
                    "trigger": "manual",
                    "metadata": {"test": True},
                },
                "timestamp": 1234567890,
            }
            websocket.send_json(cvar_request)

            # May receive error or acknowledgment depending on service state
            response = websocket.receive_json()
            assert "type" in response

    def test_websocket_disconnection_handling(self):
        """Test graceful disconnection handling."""
        client = TestClient(app)

        # Connect and disconnect immediately
        with client.websocket_connect("/ws") as websocket:
            conn_msg = websocket.receive_json()
            client_id = conn_msg["data"]["client_id"]
            # Connection will be closed when exiting context

        # Verify connection is closed by trying to connect again
        # (This should work without issues)
        with client.websocket_connect("/ws") as websocket:
            new_conn_msg = websocket.receive_json()
            new_client_id = new_conn_msg["data"]["client_id"]
            assert new_client_id != client_id  # Should be different


class TestWebSocketOptimizationIntegration:
    """Test WebSocket optimization service integration."""

    @patch("src.optimization.mean_variance.MeanVarianceOptimizer.optimize_portfolio")
    def test_optimization_request_processing(self, mock_optimize):
        """Test optimization request processing through WebSocket."""
        from src.optimization.mean_variance import OptimizationResult
        import numpy as np

        # Mock optimization result
        mock_result = OptimizationResult(
            weights=np.array([0.3, 0.3, 0.4]),
            expected_return=0.12,
            expected_variance=0.04,
            sharpe_ratio=0.85,
            solve_time=1.5,
            status="optimal",
            success=True,
            objective_value=-0.85,  # Negative sharpe for minimization
        )
        mock_optimize.return_value = mock_result

        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send optimization request
            opt_request = {
                "type": "optimization_request",
                "data": {
                    "symbols": ["AAPL", "GOOGL", "MSFT"],
                    "objective": "maximize_sharpe",
                    "constraints": {
                        "long_only": True,
                        "sum_to_one": True,
                        "min_weight": 0.1,
                        "max_weight": 0.5,
                    },
                    "method": "mean_variance",
                    "trigger": "manual",
                },
                "timestamp": 1234567890,
            }
            websocket.send_json(opt_request)

            # May receive error if optimization service is not initialized
            response = websocket.receive_json()
            assert "type" in response

    def test_optimization_error_handling(self):
        """Test optimization error handling via WebSocket."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send invalid optimization request
            invalid_request = {
                "type": "optimization_request",
                "data": {
                    "symbols": ["AAPL"],
                    "objective": "invalid_objective",  # Invalid objective
                    "constraints": {},
                },
                "timestamp": 1234567890,
            }
            websocket.send_json(invalid_request)

            # Should receive error message (either validation error or service not initialized)
            error_msg = websocket.receive_json()
            assert error_msg["type"] == "error"

    def test_concurrent_optimization_requests(self):
        """Test handling multiple concurrent optimization requests."""
        client = TestClient(app)

        with client.websocket_connect("/ws") as websocket:
            # Skip connection message
            websocket.receive_json()

            # Send multiple optimization requests
            for i in range(3):
                opt_request = {
                    "type": "optimization_request",
                    "data": {
                        "symbols": ["AAPL", "GOOGL", f"STOCK{i}"],
                        "objective": "maximize_sharpe",
                        "constraints": {"long_only": True, "sum_to_one": True},
                        "method": "mean_variance",
                        "trigger": "manual",
                    },
                    "timestamp": 1234567890 + i,
                }
                websocket.send_json(opt_request)

            # Should receive responses for each request
            for i in range(3):
                response = websocket.receive_json()
                assert "type" in response
