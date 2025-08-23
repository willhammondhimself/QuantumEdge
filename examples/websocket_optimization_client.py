"""
Example WebSocket client for real-time portfolio optimization.

This script demonstrates how to connect to the QuantumEdge WebSocket server
and request real-time portfolio optimization updates.
"""

import asyncio
import json
import websockets
from datetime import datetime
from typing import Dict, Any, List


class QuantumEdgeWebSocketClient:
    """WebSocket client for QuantumEdge real-time optimization."""

    def __init__(self, url: str = "ws://localhost:8000/ws"):
        """Initialize client with WebSocket URL."""
        self.url = url
        self.websocket = None
        self.client_id = None
        self.subscriptions = set()

    async def connect(self):
        """Connect to WebSocket server."""
        try:
            self.websocket = await websockets.connect(self.url)
            print(f"Connected to {self.url}")

            # Start listening for messages
            await self._listen()

        except Exception as e:
            print(f"Connection error: {e}")
            raise

    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket:
            await self.websocket.close()
            print("Disconnected from server")

    async def _listen(self):
        """Listen for messages from server."""
        async for message in self.websocket:
            try:
                data = json.loads(message)
                await self._handle_message(data)
            except json.JSONDecodeError as e:
                print(f"Invalid JSON received: {e}")
            except Exception as e:
                print(f"Error handling message: {e}")

    async def _handle_message(self, data: Dict[str, Any]):
        """Handle incoming message from server."""
        msg_type = data.get("type")
        msg_data = data.get("data", {})
        timestamp = data.get("timestamp")

        print(
            f"\n[{datetime.fromtimestamp(timestamp) if timestamp else 'Unknown time'}] {msg_type}"
        )

        if msg_type == "connection_status":
            self.client_id = msg_data.get("client_id")
            print(f"Connected with client ID: {self.client_id}")

        elif msg_type == "optimization_queued":
            print(f"Optimization request queued")
            print(f"  Request ID: {msg_data.get('request_id')}")
            print(f"  Queue position: {msg_data.get('queue_position')}")
            print(f"  Estimated wait: {msg_data.get('estimated_wait')} seconds")

        elif msg_type == "optimization_result":
            print(f"Optimization completed!")
            print(f"  Request ID: {msg_data.get('request_id')}")
            print(f"  Method: {msg_data.get('method')}")
            print(f"  Trigger: {msg_data.get('trigger')}")

            metrics = msg_data.get("metrics", {})
            print(f"  Expected Return: {metrics.get('expected_return', 0):.2%}")
            print(f"  Expected Volatility: {metrics.get('expected_volatility', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")

            weights = msg_data.get("weights", {})
            print(f"  Portfolio Weights:")
            for symbol, weight in weights.items():
                if weight > 0.001:  # Only show non-negligible weights
                    print(f"    {symbol}: {weight:.1%}")

            print(
                f"  Optimization Time: {msg_data.get('optimization_time', 0):.3f} seconds"
            )

        elif msg_type == "error":
            print(f"Error: {msg_data.get('error')}")
            if "detail" in msg_data:
                print(f"  Detail: {msg_data.get('detail')}")

        elif msg_type == "heartbeat":
            print(f"Heartbeat received - connection alive")

        elif msg_type == "subscribe":
            topic = msg_data.get("topic")
            if msg_data.get("status") == "subscribed":
                self.subscriptions.add(topic)
                print(f"Successfully subscribed to topic: {topic}")

        elif msg_type == "price_update":
            print(f"Price update: {msg_data}")

        elif msg_type == "portfolio_update":
            print(f"Portfolio update: {msg_data}")

        else:
            print(f"Unknown message type: {msg_type}")
            print(f"  Data: {msg_data}")

    async def send_message(self, message_type: str, data: Dict[str, Any]):
        """Send message to server."""
        if not self.websocket:
            raise RuntimeError("Not connected to server")

        message = {
            "type": message_type,
            "data": data,
            "timestamp": datetime.utcnow().timestamp(),
        }

        await self.websocket.send(json.dumps(message))
        print(f"Sent {message_type} message")

    async def request_optimization(
        self,
        symbols: List[str],
        objective: str = "maximize_sharpe",
        constraints: Dict[str, Any] = None,
        method: str = "mean_variance",
        trigger: str = "manual",
    ):
        """Request portfolio optimization."""
        if constraints is None:
            constraints = {
                "long_only": True,
                "sum_to_one": True,
                "min_weight": 0.0,
                "max_weight": 0.4,
            }

        data = {
            "symbols": symbols,
            "objective": objective,
            "constraints": constraints,
            "method": method,
            "trigger": trigger,
            "metadata": {
                "request_time": datetime.utcnow().isoformat(),
                "client_version": "1.0.0",
            },
        }

        await self.send_message("optimization_request", data)

    async def subscribe_to_topic(self, topic: str):
        """Subscribe to a topic for updates."""
        await self.send_message("subscribe", {"topic": topic})

    async def send_heartbeat(self):
        """Send heartbeat to keep connection alive."""
        await self.send_message("heartbeat", {"status": "alive"})


async def run_optimization_demo():
    """Run demonstration of WebSocket optimization."""
    client = QuantumEdgeWebSocketClient()

    try:
        # Connect to server
        connection_task = asyncio.create_task(client.connect())

        # Wait a bit for connection to establish
        await asyncio.sleep(2)

        # Subscribe to portfolio updates
        await client.subscribe_to_topic("portfolio_updates")

        # Request optimization with different methods
        print("\n=== Requesting Mean-Variance Optimization ===")
        await client.request_optimization(
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
            objective="maximize_sharpe",
            method="mean_variance",
        )

        await asyncio.sleep(5)

        print("\n=== Requesting Genetic Algorithm Optimization ===")
        await client.request_optimization(
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
            objective="minimize_variance",
            method="genetic_algorithm",
        )

        await asyncio.sleep(5)

        print("\n=== Requesting CVaR Optimization ===")
        await client.request_optimization(
            symbols=["AAPL", "GOOGL", "MSFT", "AMZN", "META"],
            objective="minimize_cvar",
            constraints={
                "long_only": True,
                "sum_to_one": True,
                "min_weight": 0.05,
                "max_weight": 0.35,
            },
            method="mean_variance",
        )

        # Keep connection open to receive results
        await asyncio.sleep(10)

        # Send heartbeat
        await client.send_heartbeat()

        # Wait for more messages
        await connection_task

    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await client.disconnect()


async def run_continuous_monitoring():
    """Run continuous portfolio monitoring with periodic optimization."""
    client = QuantumEdgeWebSocketClient()

    try:
        # Connect to server
        connection_task = asyncio.create_task(client.connect())

        await asyncio.sleep(2)

        # Subscribe to relevant topics
        await client.subscribe_to_topic("portfolio_updates")
        await client.subscribe_to_topic("price_updates")
        await client.subscribe_to_topic("risk_alerts")

        # Initial optimization
        portfolio = ["AAPL", "GOOGL", "MSFT", "NVDA", "TSLA", "JPM", "JNJ", "BRK.B"]

        await client.request_optimization(
            symbols=portfolio,
            objective="maximize_sharpe",
            constraints={
                "long_only": True,
                "sum_to_one": True,
                "min_weight": 0.02,
                "max_weight": 0.25,
            },
        )

        # Periodic optimization every 30 seconds (for demo purposes)
        while True:
            await asyncio.sleep(30)

            # Send heartbeat
            await client.send_heartbeat()

            # Request optimization with time-based trigger
            await client.request_optimization(
                symbols=portfolio, objective="maximize_sharpe", trigger="time_based"
            )

    except KeyboardInterrupt:
        print("\nShutting down monitoring...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    print("QuantumEdge WebSocket Optimization Client")
    print("=" * 50)
    print("1. Run optimization demo")
    print("2. Run continuous monitoring")

    choice = input("\nSelect option (1 or 2): ")

    if choice == "1":
        asyncio.run(run_optimization_demo())
    elif choice == "2":
        asyncio.run(run_continuous_monitoring())
    else:
        print("Invalid choice")
