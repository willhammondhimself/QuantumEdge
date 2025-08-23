/**
 * WebSocket service for real-time portfolio monitoring
 */

export interface PortfolioMessage {
  type: 'portfolio_added' | 'portfolio_removed' | 'portfolio_value_update';
  portfolio_id: string;
  timestamp: string;
  data?: any;
}

export interface MarketDataMessage {
  type: 'market_data';
  symbol: string;
  price: number;
  timestamp: string;
  change_percent?: number;
}

export interface RiskAlertMessage {
  type: 'risk_alert';
  portfolio_id: string;
  alert_type: 'volatility' | 'drawdown' | 'concentration';
  severity: 'low' | 'medium' | 'high';
  message: string;
  timestamp: string;
  metric_value: number;
  threshold: number;
}

export type WebSocketMessage = PortfolioMessage | MarketDataMessage | RiskAlertMessage;

export interface PortfolioState {
  portfolio_id: string;
  symbols: string[];
  weights: number[];
  value: number;
  last_update: string;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectInterval: number = 5000;
  private maxReconnectAttempts: number = 5;
  private reconnectAttempts: number = 0;
  private listeners: Map<string, ((message: WebSocketMessage) => void)[]> = new Map();
  private isConnecting: boolean = false;
  private portfolios: Map<string, PortfolioState> = new Map();
  private marketData: Map<string, { price: number; timestamp: string; change_percent?: number }> = new Map();

  private getWebSocketUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = process.env.NEXT_PUBLIC_API_URL?.replace(/^https?:\/\//, '') || 'localhost:8000';
    return `${protocol}//${host}/ws`;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      if (this.isConnecting) {
        // Wait for current connection attempt
        setTimeout(() => {
          if (this.ws?.readyState === WebSocket.OPEN) {
            resolve();
          } else {
            reject(new Error('Connection timeout'));
          }
        }, 5000);
        return;
      }

      this.isConnecting = true;
      
      try {
        this.ws = new WebSocket(this.getWebSocketUrl());

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message: WebSocketMessage = JSON.parse(event.data);
            this.handleMessage(message);
            this.notifyListeners(message.type, message);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error, event.data);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.isConnecting = false;
          this.ws = null;
          
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            console.log(`Attempting to reconnect (${this.reconnectAttempts + 1}/${this.maxReconnectAttempts})...`);
            setTimeout(() => {
              this.reconnectAttempts++;
              this.connect().catch(console.error);
            }, this.reconnectInterval);
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.isConnecting = false;
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnecting');
      this.ws = null;
    }
  }

  private handleMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'portfolio_added':
        if (message.data) {
          this.portfolios.set(message.portfolio_id, {
            portfolio_id: message.portfolio_id,
            symbols: message.data.symbols || [],
            weights: message.data.weights || [],
            value: message.data.value || 0,
            last_update: message.timestamp
          });
        }
        break;

      case 'portfolio_removed':
        this.portfolios.delete(message.portfolio_id);
        break;

      case 'portfolio_value_update':
        if (message.data) {
          const existing = this.portfolios.get(message.portfolio_id);
          if (existing) {
            this.portfolios.set(message.portfolio_id, {
              ...existing,
              value: message.data.value,
              last_update: message.timestamp
            });
          }
        }
        break;

      case 'market_data':
        if ('symbol' in message && 'price' in message) {
          this.marketData.set(message.symbol, {
            price: message.price,
            timestamp: message.timestamp,
            change_percent: message.change_percent
          });
        }
        break;

      case 'risk_alert':
        // Risk alerts are handled by listeners
        break;
    }
  }

  // Add portfolio for monitoring
  addPortfolio(portfolioId: string, symbols: string[], weights: number[], initialValue: number = 25000): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = {
        action: 'add_portfolio',
        portfolio_id: portfolioId,
        symbols: symbols,
        weights: weights,
        initial_value: initialValue
      };
      this.ws.send(JSON.stringify(message));
    }
  }

  // Remove portfolio from monitoring
  removePortfolio(portfolioId: string): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      const message = {
        action: 'remove_portfolio',
        portfolio_id: portfolioId
      };
      this.ws.send(JSON.stringify(message));
    }
  }

  // Event listeners
  on(eventType: string, callback: (message: WebSocketMessage) => void): void {
    if (!this.listeners.has(eventType)) {
      this.listeners.set(eventType, []);
    }
    this.listeners.get(eventType)!.push(callback);
  }

  off(eventType: string, callback: (message: WebSocketMessage) => void): void {
    if (this.listeners.has(eventType)) {
      const callbacks = this.listeners.get(eventType)!;
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  private notifyListeners(eventType: string, message: WebSocketMessage): void {
    if (this.listeners.has(eventType)) {
      this.listeners.get(eventType)!.forEach(callback => {
        try {
          callback(message);
        } catch (error) {
          console.error('Error in WebSocket listener:', error);
        }
      });
    }
  }

  // Getters for current state
  getPortfolios(): Map<string, PortfolioState> {
    return new Map(this.portfolios);
  }

  getMarketData(): Map<string, { price: number; timestamp: string; change_percent?: number }> {
    return new Map(this.marketData);
  }

  getPortfolio(portfolioId: string): PortfolioState | undefined {
    return this.portfolios.get(portfolioId);
  }

  getPrice(symbol: string): { price: number; timestamp: string; change_percent?: number } | undefined {
    return this.marketData.get(symbol);
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Export singleton instance
export const webSocketService = new WebSocketService();
export default webSocketService;