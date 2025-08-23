/**
 * React hook for WebSocket functionality
 */

import { useEffect, useState, useCallback, useRef } from 'react';
import webSocketService, { WebSocketMessage, PortfolioState } from '@/services/websocket';

interface UseWebSocketReturn {
  isConnected: boolean;
  portfolios: Map<string, PortfolioState>;
  marketData: Map<string, { price: number; timestamp: string; change_percent?: number }>;
  riskAlerts: WebSocketMessage[];
  addPortfolio: (portfolioId: string, symbols: string[], weights: number[], initialValue?: number) => void;
  removePortfolio: (portfolioId: string) => void;
  clearRiskAlerts: () => void;
  connect: () => Promise<void>;
  disconnect: () => void;
}

export function useWebSocket(): UseWebSocketReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [portfolios, setPortfolios] = useState<Map<string, PortfolioState>>(new Map());
  const [marketData, setMarketData] = useState<Map<string, { price: number; timestamp: string; change_percent?: number }>>(new Map());
  const [riskAlerts, setRiskAlerts] = useState<WebSocketMessage[]>([]);
  const mounted = useRef(true);

  // Message handlers
  const handlePortfolioMessage = useCallback((message: WebSocketMessage) => {
    if (!mounted.current) return;
    
    if (message.type === 'portfolio_added' || message.type === 'portfolio_removed' || message.type === 'portfolio_value_update') {
      setPortfolios(new Map(webSocketService.getPortfolios()));
    }
  }, []);

  const handleMarketData = useCallback((message: WebSocketMessage) => {
    if (!mounted.current) return;
    
    if (message.type === 'market_data') {
      setMarketData(new Map(webSocketService.getMarketData()));
    }
  }, []);

  const handleRiskAlert = useCallback((message: WebSocketMessage) => {
    if (!mounted.current) return;
    
    if (message.type === 'risk_alert') {
      setRiskAlerts(prev => [...prev, message].slice(-50)); // Keep last 50 alerts
    }
  }, []);

  // Connect to WebSocket
  const connect = useCallback(async () => {
    try {
      await webSocketService.connect();
      setIsConnected(webSocketService.isConnected());
      
      // Initial state sync
      setPortfolios(new Map(webSocketService.getPortfolios()));
      setMarketData(new Map(webSocketService.getMarketData()));
    } catch (error) {
      console.error('Failed to connect to WebSocket:', error);
      setIsConnected(false);
    }
  }, []);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    webSocketService.disconnect();
    setIsConnected(false);
  }, []);

  // Portfolio management
  const addPortfolio = useCallback((portfolioId: string, symbols: string[], weights: number[], initialValue = 25000) => {
    webSocketService.addPortfolio(portfolioId, symbols, weights, initialValue);
  }, []);

  const removePortfolio = useCallback((portfolioId: string) => {
    webSocketService.removePortfolio(portfolioId);
  }, []);

  const clearRiskAlerts = useCallback(() => {
    setRiskAlerts([]);
  }, []);

  // Setup event listeners
  useEffect(() => {
    // Set up event listeners
    webSocketService.on('portfolio_added', handlePortfolioMessage);
    webSocketService.on('portfolio_removed', handlePortfolioMessage);
    webSocketService.on('portfolio_value_update', handlePortfolioMessage);
    webSocketService.on('market_data', handleMarketData);
    webSocketService.on('risk_alert', handleRiskAlert);

    // Connection state monitoring
    const checkConnection = () => {
      if (mounted.current) {
        setIsConnected(webSocketService.isConnected());
      }
    };

    const connectionInterval = setInterval(checkConnection, 1000);

    return () => {
      // Cleanup
      webSocketService.off('portfolio_added', handlePortfolioMessage);
      webSocketService.off('portfolio_removed', handlePortfolioMessage);
      webSocketService.off('portfolio_value_update', handlePortfolioMessage);
      webSocketService.off('market_data', handleMarketData);
      webSocketService.off('risk_alert', handleRiskAlert);
      clearInterval(connectionInterval);
    };
  }, [handlePortfolioMessage, handleMarketData, handleRiskAlert]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      mounted.current = false;
    };
  }, []);

  return {
    isConnected,
    portfolios,
    marketData,
    riskAlerts,
    addPortfolio,
    removePortfolio,
    clearRiskAlerts,
    connect,
    disconnect
  };
}

export default useWebSocket;