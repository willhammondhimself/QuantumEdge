#!/usr/bin/env python3
"""
Test script for WebSocket real-time portfolio monitoring.

This script demonstrates the WebSocket functionality by:
1. Starting a test portfolio monitoring session
2. Connecting to the WebSocket endpoint
3. Subscribing to market data and portfolio updates
4. Displaying real-time updates
"""

import asyncio
import json
import websockets
import requests
from datetime import datetime


async def test_websocket_connection():
    """Test WebSocket connection and real-time updates."""
    
    print("🚀 Testing QuantumEdge WebSocket Real-Time Portfolio Monitoring")
    print("=" * 60)
    
    # First, add a test portfolio for monitoring
    portfolio_data = {
        "portfolio_id": "test-portfolio-001",
        "name": "Test Portfolio",
        "holdings": {
            "AAPL": 10.0,
            "GOOGL": 5.0,
            "MSFT": 15.0,
            "TSLA": 8.0
        },
        "initial_value": 50000.0,
        "benchmark_symbol": "SPY",
        "risk_free_rate": 0.02
    }
    
    try:
        # Add portfolio to monitoring (assumes API is running on port 8000)
        response = requests.post(
            "http://localhost:8000/api/v1/streaming/portfolio",
            json=portfolio_data,
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Portfolio added for monitoring")
            print(f"   Portfolio ID: {portfolio_data['portfolio_id']}")
            print(f"   Holdings: {list(portfolio_data['holdings'].keys())}")
        else:
            print(f"❌ Failed to add portfolio: {response.status_code}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to API server: {e}")
        print("💡 Make sure the API server is running: python -m src.api.main")
        return
    
    # Connect to WebSocket
    uri = "ws://localhost:8000/ws"
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"🔌 Connected to WebSocket: {uri}")
            
            # Wait for connection confirmation
            response = await websocket.recv()
            data = json.loads(response)
            print(f"📡 {data['type']}: {data['data']}")
            
            # Subscribe to market data
            subscribe_message = {
                "type": "subscribe",
                "data": {"topic": "market_data:all"}
            }
            await websocket.send(json.dumps(subscribe_message))
            print("📊 Subscribed to market data updates")
            
            # Subscribe to portfolio updates
            portfolio_subscribe = {
                "type": "subscribe", 
                "data": {"topic": f"portfolio:{portfolio_data['portfolio_id']}"}
            }
            await websocket.send(json.dumps(portfolio_subscribe))
            print("💼 Subscribed to portfolio updates")
            
            # Subscribe to alerts
            alert_subscribe = {
                "type": "subscribe",
                "data": {"topic": "alerts"}
            }
            await websocket.send(json.dumps(alert_subscribe))
            print("🚨 Subscribed to alerts")
            
            print("\n📈 Listening for real-time updates (press Ctrl+C to stop)...")
            print("-" * 60)
            
            # Listen for messages
            message_count = 0
            while message_count < 50:  # Limit for demo
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                    data = json.loads(response)
                    message_count += 1
                    
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    if data['type'] == 'price_update':
                        symbol = data['data']['symbol']
                        price = data['data']['price']
                        change_pct = data['data']['change_percent']
                        print(f"[{timestamp}] 📊 {symbol}: ${price:.2f} ({change_pct:+.2%})")
                    
                    elif data['type'] == 'portfolio_update':
                        portfolio_id = data['data']['portfolio_id']
                        total_value = data['data']['total_value']
                        change_pct = data['data']['change_percent']
                        print(f"[{timestamp}] 💼 Portfolio {portfolio_id}: ${total_value:,.2f} ({change_pct:+.2%})")
                    
                    elif data['type'] == 'risk_update':
                        portfolio_id = data['data']['portfolio_id']
                        volatility = data['data']['volatility']
                        sharpe = data['data']['sharpe_ratio']
                        print(f"[{timestamp}] ⚠️  Risk {portfolio_id}: Vol={volatility:.2%}, Sharpe={sharpe:.3f}")
                    
                    elif data['type'] == 'alert':
                        alert_msg = data['data']['message']
                        severity = data['data']['severity']
                        print(f"[{timestamp}] 🚨 {severity.upper()}: {alert_msg}")
                    
                    elif data['type'] == 'heartbeat':
                        print(f"[{timestamp}] 💓 Heartbeat")
                    
                    else:
                        print(f"[{timestamp}] 📨 {data['type']}: {data['data']}")
                
                except asyncio.TimeoutError:
                    # Send heartbeat
                    heartbeat = {"type": "heartbeat", "data": {"client_time": datetime.now().isoformat()}}
                    await websocket.send(json.dumps(heartbeat))
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] 💓 Sent heartbeat")
            
            print(f"\n✅ Received {message_count} messages successfully")
            
    except websockets.exceptions.ConnectionClosed:
        print("🔌 WebSocket connection closed")
    except websockets.exceptions.WebSocketException as e:
        print(f"❌ WebSocket error: {e}")
    except KeyboardInterrupt:
        print("\n⏹️  Stopped by user")
    
    # Cleanup - remove test portfolio
    try:
        delete_response = requests.delete(
            f"http://localhost:8000/api/v1/streaming/portfolio/{portfolio_data['portfolio_id']}",
            timeout=5
        )
        if delete_response.status_code == 200:
            print("🧹 Test portfolio removed from monitoring")
    except:
        pass  # Ignore cleanup errors
    
    print("\n🎉 WebSocket test completed!")


if __name__ == "__main__":
    print("QuantumEdge WebSocket Test")
    print("Make sure the API server is running first:")
    print("  python -m src.api.main")
    print()
    
    try:
        asyncio.run(test_websocket_connection())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")