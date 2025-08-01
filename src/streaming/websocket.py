"""
WebSocket connection manager for real-time portfolio monitoring.

This module provides WebSocket connection management, message routing,
and real-time data streaming capabilities for the QuantumEdge platform.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket message types."""
    PORTFOLIO_UPDATE = "portfolio_update"
    PRICE_UPDATE = "price_update"
    RISK_UPDATE = "risk_update"
    ALERT = "alert"
    HEARTBEAT = "heartbeat"
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    ERROR = "error"
    CONNECTION_STATUS = "connection_status"


@dataclass
class WebSocketMessage:
    """WebSocket message structure."""
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: float = None
    client_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps({
            "type": self.message_type.value,
            "data": self.data,
            "timestamp": self.timestamp,
            "client_id": self.client_id
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'WebSocketMessage':
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls(
            message_type=MessageType(data["type"]),
            data=data["data"],
            timestamp=data.get("timestamp"),
            client_id=data.get("client_id")
        )


@dataclass
class ClientConnection:
    """WebSocket client connection information."""
    client_id: str
    websocket: WebSocket
    subscriptions: Set[str]
    connected_at: datetime
    last_heartbeat: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()


class WebSocketConnectionManager:
    """
    Manages WebSocket connections and message broadcasting.
    
    Provides connection lifecycle management, subscription handling,
    and message routing for real-time portfolio monitoring.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize WebSocket connection manager.
        
        Args:
            redis_client: Optional Redis client for pub/sub messaging
        """
        self.connections: Dict[str, ClientConnection] = {}
        self.subscriptions: Dict[str, Set[str]] = {}  # topic -> set of client_ids
        self.redis_client = redis_client
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 60  # seconds
        
        # Start background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._tasks_started = False
    
    async def _setup_background_tasks(self):
        """Setup background tasks for connection management."""
        if self._tasks_started:
            return
        
        # Heartbeat monitoring
        heartbeat_task = asyncio.create_task(self._heartbeat_monitor())
        self._background_tasks.append(heartbeat_task)
        
        # Redis pub/sub listener (if Redis is available)
        if self.redis_client:
            pubsub_task = asyncio.create_task(self._redis_pubsub_listener())
            self._background_tasks.append(pubsub_task)
        
        self._tasks_started = True
        logger.info("WebSocket background tasks started")
    
    async def connect(
        self, 
        websocket: WebSocket, 
        client_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Accept WebSocket connection and register client.
        
        Args:
            websocket: WebSocket connection
            client_id: Optional client identifier
            metadata: Optional client metadata
            
        Returns:
            Client ID for the connection
        """
        await websocket.accept()
        
        # Setup background tasks if not already started
        await self._setup_background_tasks()
        
        if client_id is None:
            client_id = str(uuid.uuid4())
        
        connection = ClientConnection(
            client_id=client_id,
            websocket=websocket,
            subscriptions=set(),
            connected_at=datetime.utcnow(),
            last_heartbeat=time.time(),
            metadata=metadata or {}
        )
        
        self.connections[client_id] = connection
        
        # Send connection confirmation
        await self.send_to_client(
            client_id,
            WebSocketMessage(
                message_type=MessageType.CONNECTION_STATUS,
                data={
                    "status": "connected",
                    "client_id": client_id,
                    "server_time": datetime.utcnow().isoformat()
                },
                client_id=client_id
            )
        )
        
        logger.info(f"WebSocket client connected: {client_id}")
        return client_id
    
    async def disconnect(self, client_id: str):
        """
        Disconnect and cleanup client connection.
        
        Args:
            client_id: Client identifier to disconnect
        """
        if client_id not in self.connections:
            return
        
        connection = self.connections[client_id]
        
        # Remove from all subscriptions
        for topic in connection.subscriptions.copy():
            await self.unsubscribe(client_id, topic)
        
        # Close WebSocket connection
        try:
            await connection.websocket.close()
        except Exception as e:
            logger.warning(f"Error closing WebSocket for {client_id}: {e}")
        
        # Remove from connections
        del self.connections[client_id]
        logger.info(f"WebSocket client disconnected: {client_id}")
    
    async def send_to_client(self, client_id: str, message: WebSocketMessage):
        """
        Send message to specific client.
        
        Args:
            client_id: Target client identifier
            message: Message to send
        """
        if client_id not in self.connections:
            logger.warning(f"Attempted to send message to non-existent client: {client_id}")
            return
        
        connection = self.connections[client_id]
        message.client_id = client_id
        
        try:
            await connection.websocket.send_text(message.to_json())
        except WebSocketDisconnect:
            logger.info(f"Client {client_id} disconnected during message send")
            await self.disconnect(client_id)
        except Exception as e:
            logger.error(f"Error sending message to {client_id}: {e}")
            await self.disconnect(client_id)
    
    async def broadcast_to_topic(self, topic: str, message: WebSocketMessage):
        """
        Broadcast message to all clients subscribed to a topic.
        
        Args:
            topic: Topic name
            message: Message to broadcast
        """
        if topic not in self.subscriptions:
            return
        
        # Send to all subscribed clients
        tasks = []
        for client_id in self.subscriptions[topic].copy():
            tasks.append(self.send_to_client(client_id, message))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def subscribe(self, client_id: str, topic: str):
        """
        Subscribe client to a topic.
        
        Args:
            client_id: Client identifier
            topic: Topic to subscribe to
        """
        if client_id not in self.connections:
            logger.warning(f"Attempted to subscribe non-existent client: {client_id}")
            return
        
        # Add to client's subscriptions
        self.connections[client_id].subscriptions.add(topic)
        
        # Add to topic subscriptions
        if topic not in self.subscriptions:
            self.subscriptions[topic] = set()
        self.subscriptions[topic].add(client_id)
        
        logger.info(f"Client {client_id} subscribed to topic: {topic}")
        
        # Send subscription confirmation
        await self.send_to_client(
            client_id,
            WebSocketMessage(
                message_type=MessageType.SUBSCRIBE,
                data={"topic": topic, "status": "subscribed"},
                client_id=client_id
            )
        )
    
    async def unsubscribe(self, client_id: str, topic: str):
        """
        Unsubscribe client from a topic.
        
        Args:
            client_id: Client identifier
            topic: Topic to unsubscribe from
        """
        if client_id not in self.connections:
            return
        
        # Remove from client's subscriptions
        self.connections[client_id].subscriptions.discard(topic)
        
        # Remove from topic subscriptions
        if topic in self.subscriptions:
            self.subscriptions[topic].discard(client_id)
            
            # Clean up empty topic
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
        
        logger.info(f"Client {client_id} unsubscribed from topic: {topic}")
        
        # Send unsubscription confirmation
        await self.send_to_client(
            client_id,
            WebSocketMessage(
                message_type=MessageType.UNSUBSCRIBE,
                data={"topic": topic, "status": "unsubscribed"},
                client_id=client_id
            )
        )
    
    async def handle_message(self, client_id: str, message_data: str):
        """
        Handle incoming message from client.
        
        Args:
            client_id: Client identifier
            message_data: Raw message data
        """
        try:
            message = WebSocketMessage.from_json(message_data)
            
            # Update last heartbeat
            if client_id in self.connections:
                self.connections[client_id].last_heartbeat = time.time()
            
            # Handle different message types
            if message.message_type == MessageType.SUBSCRIBE:
                topic = message.data.get("topic")
                if topic:
                    await self.subscribe(client_id, topic)
            
            elif message.message_type == MessageType.UNSUBSCRIBE:
                topic = message.data.get("topic")
                if topic:
                    await self.unsubscribe(client_id, topic)
            
            elif message.message_type == MessageType.HEARTBEAT:
                # Respond with heartbeat
                await self.send_to_client(
                    client_id,
                    WebSocketMessage(
                        message_type=MessageType.HEARTBEAT,
                        data={"status": "alive", "server_time": time.time()},
                        client_id=client_id
                    )
                )
            
            else:
                logger.warning(f"Unknown message type from {client_id}: {message.message_type}")
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client {client_id}: {e}")
            await self.send_to_client(
                client_id,
                WebSocketMessage(
                    message_type=MessageType.ERROR,
                    data={"error": "Invalid JSON format"},
                    client_id=client_id
                )
            )
        except Exception as e:
            logger.error(f"Error handling message from {client_id}: {e}")
            await self.send_to_client(
                client_id,
                WebSocketMessage(
                    message_type=MessageType.ERROR,
                    data={"error": "Message processing failed"},
                    client_id=client_id
                )
            )
    
    async def _heartbeat_monitor(self):
        """Monitor client heartbeats and disconnect stale connections."""
        while True:
            try:
                current_time = time.time()
                stale_clients = []
                
                for client_id, connection in self.connections.items():
                    if current_time - connection.last_heartbeat > self.connection_timeout:
                        stale_clients.append(client_id)
                
                # Disconnect stale clients
                for client_id in stale_clients:
                    logger.info(f"Disconnecting stale client: {client_id}")
                    await self.disconnect(client_id)
                
                await asyncio.sleep(self.heartbeat_interval)
            
            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _redis_pubsub_listener(self):
        """Listen for Redis pub/sub messages for distributed messaging."""
        if not self.redis_client:
            return
        
        try:
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("quantumedge:websocket:*")
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        channel = message["channel"].decode()
                        topic = channel.split(":")[-1]
                        data = json.loads(message["data"].decode())
                        
                        ws_message = WebSocketMessage(
                            message_type=MessageType(data["type"]),
                            data=data["data"],
                            timestamp=data.get("timestamp")
                        )
                        
                        await self.broadcast_to_topic(topic, ws_message)
                    
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
        
        except Exception as e:
            logger.error(f"Redis pub/sub listener error: {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.connections),
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "topics": list(self.subscriptions.keys()),
            "clients": list(self.connections.keys())
        }
    
    async def cleanup(self):
        """Cleanup resources and background tasks."""
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Disconnect all clients
        for client_id in list(self.connections.keys()):
            await self.disconnect(client_id)
        
        logger.info("WebSocket connection manager cleaned up")


# Global connection manager instance
connection_manager = WebSocketConnectionManager()