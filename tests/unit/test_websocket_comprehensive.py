"""Comprehensive tests for WebSocket connection manager."""

import pytest
import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock, call
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from src.streaming.websocket import (
    MessageType,
    WebSocketMessage,
    ClientConnection,
    WebSocketConnectionManager,
)


class TestMessageType:
    """Test MessageType enum."""

    def test_message_types(self):
        """Test all message types are defined."""
        assert MessageType.PORTFOLIO_UPDATE == "portfolio_update"
        assert MessageType.PRICE_UPDATE == "price_update"
        assert MessageType.RISK_UPDATE == "risk_update"
        assert MessageType.ALERT == "alert"
        assert MessageType.HEARTBEAT == "heartbeat"
        assert MessageType.SUBSCRIBE == "subscribe"
        assert MessageType.UNSUBSCRIBE == "unsubscribe"
        assert MessageType.ERROR == "error"
        assert MessageType.CONNECTION_STATUS == "connection_status"
        assert MessageType.OPTIMIZATION_REQUEST == "optimization_request"
        assert MessageType.OPTIMIZATION_RESULT == "optimization_result"
        assert MessageType.OPTIMIZATION_STATUS == "optimization_status"


class TestWebSocketMessage:
    """Test WebSocketMessage dataclass."""

    def test_message_creation(self):
        """Test creating a WebSocket message."""
        message = WebSocketMessage(
            message_type=MessageType.PORTFOLIO_UPDATE,
            data={"portfolio_id": "port_123", "value": 100000},
            client_id="client_456",
        )

        assert message.message_type == MessageType.PORTFOLIO_UPDATE
        assert message.data["portfolio_id"] == "port_123"
        assert message.client_id == "client_456"
        assert message.timestamp is not None
        assert isinstance(message.timestamp, float)

    def test_message_with_timestamp(self):
        """Test creating message with explicit timestamp."""
        timestamp = time.time()
        message = WebSocketMessage(
            message_type=MessageType.ALERT,
            data={"alert": "High volatility"},
            timestamp=timestamp,
        )

        assert message.timestamp == timestamp

    def test_message_to_json(self):
        """Test converting message to JSON."""
        message = WebSocketMessage(
            message_type=MessageType.PRICE_UPDATE,
            data={"symbol": "AAPL", "price": 150.0},
            client_id="client_123",
        )

        json_str = message.to_json()
        data = json.loads(json_str)

        assert data["type"] == "price_update"
        assert data["data"]["symbol"] == "AAPL"
        assert data["data"]["price"] == 150.0
        assert data["client_id"] == "client_123"
        assert "timestamp" in data

    def test_message_from_json(self):
        """Test creating message from JSON."""
        json_str = json.dumps(
            {
                "type": "risk_update",
                "data": {"volatility": 0.25, "var": -0.05},
                "timestamp": 1234567890.0,
                "client_id": "client_789",
            }
        )

        message = WebSocketMessage.from_json(json_str)

        assert message.message_type == MessageType.RISK_UPDATE
        assert message.data["volatility"] == 0.25
        assert message.timestamp == 1234567890.0
        assert message.client_id == "client_789"

    def test_message_from_json_minimal(self):
        """Test creating message from minimal JSON."""
        json_str = json.dumps({"type": "heartbeat", "data": {}})

        message = WebSocketMessage.from_json(json_str)

        assert message.message_type == MessageType.HEARTBEAT
        assert message.data == {}
        # Timestamp gets auto-set in __post_init__ if not provided
        assert message.timestamp is not None
        assert message.client_id is None


class TestClientConnection:
    """Test ClientConnection dataclass."""

    def test_client_connection_creation(self):
        """Test creating client connection."""
        mock_websocket = Mock(spec=WebSocket)
        connected_at = datetime.utcnow()

        connection = ClientConnection(
            client_id="client_123",
            websocket=mock_websocket,
            subscriptions={"portfolio_updates", "price_updates"},
            connected_at=connected_at,
            last_heartbeat=None,
            metadata={"user_id": "user_456"},
        )

        assert connection.client_id == "client_123"
        assert connection.websocket == mock_websocket
        assert "portfolio_updates" in connection.subscriptions
        assert connection.connected_at == connected_at
        assert connection.last_heartbeat is not None  # Auto-set in __post_init__
        assert connection.metadata["user_id"] == "user_456"


class TestWebSocketConnectionManager:
    """Test WebSocketConnectionManager class."""

    @pytest.fixture
    def manager(self):
        """Create connection manager instance."""
        return WebSocketConnectionManager()

    @pytest.fixture
    def manager_with_redis(self):
        """Create connection manager with mock Redis."""
        mock_redis = AsyncMock(spec=redis.Redis)
        return WebSocketConnectionManager(redis_client=mock_redis)

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        mock_ws = AsyncMock(spec=WebSocket)
        mock_ws.accept = AsyncMock()
        mock_ws.send_text = AsyncMock()
        mock_ws.close = AsyncMock()
        return mock_ws

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert len(manager.connections) == 0
        assert len(manager.subscriptions) == 0
        assert manager.redis_client is None
        assert manager.heartbeat_interval == 30
        assert manager.connection_timeout == 60
        assert manager.optimization_handler is None
        assert manager._tasks_started is False

    @pytest.mark.asyncio
    async def test_connect_new_client(self, manager, mock_websocket):
        """Test connecting a new client."""
        # Connect client
        client_id = await manager.connect(mock_websocket)

        # Check WebSocket accepted
        mock_websocket.accept.assert_called_once()

        # Check client registered
        assert client_id in manager.connections
        connection = manager.connections[client_id]
        assert connection.websocket == mock_websocket
        assert len(connection.subscriptions) == 0

        # Check connection status sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "connection_status"
        assert sent_data["data"]["status"] == "connected"
        assert sent_data["data"]["client_id"] == client_id

    @pytest.mark.asyncio
    async def test_connect_with_custom_id(self, manager, mock_websocket):
        """Test connecting with custom client ID."""
        custom_id = "custom_client_123"
        metadata = {"user_id": "user_456", "role": "trader"}

        client_id = await manager.connect(mock_websocket, custom_id, metadata)

        assert client_id == custom_id
        assert client_id in manager.connections
        assert manager.connections[client_id].metadata == metadata

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """Test disconnecting a client."""
        # Connect client
        client_id = await manager.connect(mock_websocket)

        # Subscribe to some topics
        await manager.subscribe(client_id, "topic1")
        await manager.subscribe(client_id, "topic2")

        # Disconnect
        await manager.disconnect(client_id)

        # Check client removed
        assert client_id not in manager.connections

        # Check unsubscribed from topics
        assert "topic1" not in manager.subscriptions
        assert "topic2" not in manager.subscriptions

        # Check WebSocket closed
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_client(self, manager):
        """Test disconnecting non-existent client."""
        # Should not raise error
        await manager.disconnect("nonexistent_client")

    @pytest.mark.asyncio
    async def test_send_to_client(self, manager, mock_websocket):
        """Test sending message to client."""
        # Connect client
        client_id = await manager.connect(mock_websocket)
        mock_websocket.reset_mock()

        # Send message
        message = WebSocketMessage(
            message_type=MessageType.PORTFOLIO_UPDATE, data={"value": 100000}
        )
        await manager.send_to_client(client_id, message)

        # Check message sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "portfolio_update"
        assert sent_data["data"]["value"] == 100000
        assert sent_data["client_id"] == client_id

    @pytest.mark.asyncio
    async def test_send_to_nonexistent_client(self, manager):
        """Test sending to non-existent client."""
        message = WebSocketMessage(
            message_type=MessageType.ALERT, data={"alert": "test"}
        )

        # Should not raise error
        await manager.send_to_client("nonexistent", message)

    @pytest.mark.asyncio
    async def test_send_with_disconnect_error(self, manager, mock_websocket):
        """Test handling disconnect during send."""
        # Connect client
        client_id = await manager.connect(mock_websocket)

        # Mock disconnect error
        mock_websocket.send_text.side_effect = WebSocketDisconnect()

        # Send message
        message = WebSocketMessage(
            message_type=MessageType.ALERT, data={"alert": "test"}
        )
        await manager.send_to_client(client_id, message)

        # Check client disconnected
        assert client_id not in manager.connections

    @pytest.mark.asyncio
    async def test_broadcast_to_topic(self, manager):
        """Test broadcasting to topic subscribers."""
        # Create multiple mock websockets
        mock_ws1 = AsyncMock(spec=WebSocket)
        mock_ws1.accept = AsyncMock()
        mock_ws1.send_text = AsyncMock()

        mock_ws2 = AsyncMock(spec=WebSocket)
        mock_ws2.accept = AsyncMock()
        mock_ws2.send_text = AsyncMock()

        # Connect clients
        client1 = await manager.connect(mock_ws1)
        client2 = await manager.connect(mock_ws2)

        # Subscribe to topic
        await manager.subscribe(client1, "market_updates")
        await manager.subscribe(client2, "market_updates")

        # Reset mocks
        mock_ws1.reset_mock()
        mock_ws2.reset_mock()

        # Broadcast message
        message = WebSocketMessage(
            message_type=MessageType.PRICE_UPDATE,
            data={"symbol": "AAPL", "price": 150.0},
        )
        await manager.broadcast_to_topic("market_updates", message)

        # Check both clients received message
        assert mock_ws1.send_text.call_count == 1
        assert mock_ws2.send_text.call_count == 1

    @pytest.mark.asyncio
    async def test_broadcast_to_empty_topic(self, manager):
        """Test broadcasting to topic with no subscribers."""
        message = WebSocketMessage(
            message_type=MessageType.ALERT, data={"alert": "test"}
        )

        # Should not raise error
        await manager.broadcast_to_topic("empty_topic", message)

    @pytest.mark.asyncio
    async def test_subscribe(self, manager, mock_websocket):
        """Test subscribing to topic."""
        # Connect client
        client_id = await manager.connect(mock_websocket)
        mock_websocket.reset_mock()

        # Subscribe to topic
        await manager.subscribe(client_id, "portfolio_updates")

        # Check subscription recorded
        assert "portfolio_updates" in manager.connections[client_id].subscriptions
        assert "portfolio_updates" in manager.subscriptions
        assert client_id in manager.subscriptions["portfolio_updates"]

        # Check confirmation sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "subscribe"
        assert sent_data["data"]["topic"] == "portfolio_updates"
        assert sent_data["data"]["status"] == "subscribed"

    @pytest.mark.asyncio
    async def test_subscribe_nonexistent_client(self, manager):
        """Test subscribing with non-existent client."""
        # Should not raise error
        await manager.subscribe("nonexistent", "topic")

    @pytest.mark.asyncio
    async def test_unsubscribe(self, manager, mock_websocket):
        """Test unsubscribing from topic."""
        # Connect and subscribe
        client_id = await manager.connect(mock_websocket)
        await manager.subscribe(client_id, "topic1")
        mock_websocket.reset_mock()

        # Unsubscribe
        await manager.unsubscribe(client_id, "topic1")

        # Check unsubscribed
        assert "topic1" not in manager.connections[client_id].subscriptions
        assert "topic1" not in manager.subscriptions

        # Check confirmation sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "unsubscribe"
        assert sent_data["data"]["topic"] == "topic1"
        assert sent_data["data"]["status"] == "unsubscribed"

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, manager, mock_websocket):
        """Test handling subscribe message from client."""
        # Connect client
        client_id = await manager.connect(mock_websocket)
        mock_websocket.reset_mock()

        # Create subscribe message
        message = json.dumps({"type": "subscribe", "data": {"topic": "risk_updates"}})

        # Handle message
        await manager.handle_message(client_id, message)

        # Check subscribed
        assert "risk_updates" in manager.connections[client_id].subscriptions

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(self, manager, mock_websocket):
        """Test handling unsubscribe message from client."""
        # Connect and subscribe
        client_id = await manager.connect(mock_websocket)
        await manager.subscribe(client_id, "alerts")
        mock_websocket.reset_mock()

        # Create unsubscribe message
        message = json.dumps({"type": "unsubscribe", "data": {"topic": "alerts"}})

        # Handle message
        await manager.handle_message(client_id, message)

        # Check unsubscribed
        assert "alerts" not in manager.connections[client_id].subscriptions

    @pytest.mark.asyncio
    async def test_handle_heartbeat_message(self, manager, mock_websocket):
        """Test handling heartbeat message."""
        # Connect client
        client_id = await manager.connect(mock_websocket)
        mock_websocket.reset_mock()

        # Create heartbeat message
        message = json.dumps({"type": "heartbeat", "data": {}})

        # Handle message
        await manager.handle_message(client_id, message)

        # Check heartbeat response sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "heartbeat"
        assert sent_data["data"]["status"] == "alive"

    @pytest.mark.asyncio
    async def test_handle_optimization_request(self, manager, mock_websocket):
        """Test handling optimization request."""
        # Connect client
        client_id = await manager.connect(mock_websocket)

        # Set optimization handler
        mock_handler = AsyncMock()
        manager.optimization_handler = mock_handler

        # Create optimization request
        message = json.dumps(
            {
                "type": "optimization_request",
                "data": {"symbols": ["AAPL", "GOOGL"], "objective": "maximize_sharpe"},
            }
        )

        # Handle message
        await manager.handle_message(client_id, message)

        # Check handler called
        mock_handler.assert_called_once_with(
            client_id, {"symbols": ["AAPL", "GOOGL"], "objective": "maximize_sharpe"}
        )

    @pytest.mark.asyncio
    async def test_handle_optimization_request_no_handler(
        self, manager, mock_websocket
    ):
        """Test optimization request without handler."""
        # Connect client
        client_id = await manager.connect(mock_websocket)
        mock_websocket.reset_mock()

        # Create optimization request
        message = json.dumps(
            {"type": "optimization_request", "data": {"symbols": ["AAPL"]}}
        )

        # Handle message
        await manager.handle_message(client_id, message)

        # Check error sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "not available" in sent_data["data"]["error"]

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, manager, mock_websocket):
        """Test handling invalid JSON message."""
        # Connect client
        client_id = await manager.connect(mock_websocket)
        mock_websocket.reset_mock()

        # Handle invalid JSON
        await manager.handle_message(client_id, "invalid json {")

        # Check error sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "Invalid JSON" in sent_data["data"]["error"]

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(self, manager, mock_websocket):
        """Test handling unknown message type."""
        # Connect client
        client_id = await manager.connect(mock_websocket)
        mock_websocket.reset_mock()

        # Create message with unknown type - this will raise ValueError
        message = json.dumps({"type": "unknown_type", "data": {}})

        # Handle message - will trigger error handling
        await manager.handle_message(client_id, message)

        # Check error response sent
        mock_websocket.send_text.assert_called_once()
        sent_data = json.loads(mock_websocket.send_text.call_args[0][0])
        assert sent_data["type"] == "error"
        assert "Message processing failed" in sent_data["data"]["error"]

    @pytest.mark.asyncio
    async def test_heartbeat_monitor(self, manager, mock_websocket):
        """Test heartbeat monitoring."""
        # Connect client
        client_id = await manager.connect(mock_websocket)

        # Set very short timeout for testing
        manager.connection_timeout = 0.1
        manager.heartbeat_interval = 0.05

        # Start heartbeat monitor
        monitor_task = asyncio.create_task(manager._heartbeat_monitor())

        # Wait for timeout
        await asyncio.sleep(0.2)

        # Stop monitor
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

        # Check client disconnected due to timeout
        assert client_id not in manager.connections

    @pytest.mark.asyncio
    async def test_redis_pubsub_listener(self, manager_with_redis):
        """Test Redis pub/sub listener."""
        manager = manager_with_redis

        # Mock pub/sub
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()

        # Create mock message
        mock_message = {
            "type": "message",
            "channel": b"quantumedge:websocket:market_updates",
            "data": json.dumps(
                {
                    "type": "price_update",
                    "data": {"symbol": "AAPL", "price": 150.0},
                    "timestamp": time.time(),
                }
            ).encode(),
        }

        # Mock listen to return one message then stop
        async def mock_listen():
            yield mock_message

        mock_pubsub.listen = mock_listen
        manager.redis_client.pubsub = Mock(return_value=mock_pubsub)

        # Mock broadcast_to_topic
        manager.broadcast_to_topic = AsyncMock()

        # Run listener briefly
        listener_task = asyncio.create_task(manager._redis_pubsub_listener())
        await asyncio.sleep(0.1)
        listener_task.cancel()
        try:
            await listener_task
        except asyncio.CancelledError:
            pass

        # Check message was broadcast
        manager.broadcast_to_topic.assert_called_once()
        call_args = manager.broadcast_to_topic.call_args
        assert call_args[0][0] == "market_updates"
        assert call_args[0][1].message_type == MessageType.PRICE_UPDATE

    def test_get_connection_stats(self, manager):
        """Test getting connection statistics."""
        # Add some mock data
        manager.connections["client1"] = Mock()
        manager.connections["client2"] = Mock()
        manager.subscriptions["topic1"] = {"client1", "client2"}
        manager.subscriptions["topic2"] = {"client1"}

        stats = manager.get_connection_stats()

        assert stats["total_connections"] == 2
        assert stats["total_subscriptions"] == 3  # 2 + 1
        assert "topic1" in stats["topics"]
        assert "topic2" in stats["topics"]
        assert "client1" in stats["clients"]
        assert "client2" in stats["clients"]

    @pytest.mark.asyncio
    async def test_cleanup(self, manager, mock_websocket):
        """Test cleanup method."""
        # Connect clients
        client1 = await manager.connect(mock_websocket)
        client2 = await manager.connect(AsyncMock(spec=WebSocket))

        # Ensure background tasks are started
        assert manager._tasks_started
        assert len(manager._background_tasks) > 0

        # Cleanup
        await manager.cleanup()

        # Check all clients disconnected
        assert len(manager.connections) == 0

        # Check tasks cancelled
        for task in manager._background_tasks:
            assert task.cancelled()

    @pytest.mark.asyncio
    async def test_background_tasks_start_once(self, manager):
        """Test background tasks only start once."""
        # Setup should start tasks
        await manager._setup_background_tasks()
        task_count = len(manager._background_tasks)

        # Calling again should not add more tasks
        await manager._setup_background_tasks()
        assert len(manager._background_tasks) == task_count


class TestWebSocketIntegration:
    """Integration tests for WebSocket system."""

    @pytest.mark.asyncio
    async def test_full_client_lifecycle(self):
        """Test complete client lifecycle."""
        manager = WebSocketConnectionManager()
        mock_ws = AsyncMock(spec=WebSocket)

        # Connect
        client_id = await manager.connect(mock_ws, metadata={"user": "test"})
        assert client_id in manager.connections

        # Subscribe to topics
        await manager.subscribe(client_id, "portfolio_updates")
        await manager.subscribe(client_id, "price_updates")

        # Send some messages
        await manager.send_to_client(
            client_id,
            WebSocketMessage(
                message_type=MessageType.PORTFOLIO_UPDATE, data={"value": 100000}
            ),
        )

        # Handle incoming message
        heartbeat = json.dumps({"type": "heartbeat", "data": {}})
        await manager.handle_message(client_id, heartbeat)

        # Unsubscribe from one topic
        await manager.unsubscribe(client_id, "price_updates")

        # Disconnect
        await manager.disconnect(client_id)
        assert client_id not in manager.connections

        # Cleanup
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_multiple_clients_broadcast(self):
        """Test broadcasting to multiple clients."""
        manager = WebSocketConnectionManager()

        # Connect three clients
        clients = []
        for i in range(3):
            mock_ws = AsyncMock(spec=WebSocket)
            client_id = await manager.connect(mock_ws)
            clients.append((client_id, mock_ws))

        # Subscribe all to same topic
        topic = "market_updates"
        for client_id, _ in clients:
            await manager.subscribe(client_id, topic)

        # Broadcast message
        message = WebSocketMessage(
            message_type=MessageType.PRICE_UPDATE,
            data={"symbol": "AAPL", "price": 150.0},
        )
        await manager.broadcast_to_topic(topic, message)

        # Check all clients received message
        for _, mock_ws in clients:
            assert mock_ws.send_text.call_count >= 2  # Connection + broadcast

        # Cleanup
        await manager.cleanup()
