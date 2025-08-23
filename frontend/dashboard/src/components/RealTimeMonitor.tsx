'use client';

import { useState, useEffect } from 'react';
import {
  Activity,
  TrendingUp,
  TrendingDown,
  DollarSign,
  AlertTriangle,
  Wifi,
  WifiOff,
  Play,
  Square,
  RefreshCw,
  X,
  Bell,
  BellOff
} from 'lucide-react';

import useWebSocket from '@/hooks/useWebSocket';
import { SAMPLE_SYMBOLS, ASSET_CATEGORIES, ALL_SYMBOLS } from '@/services/api';

interface MonitoredPortfolio {
  id: string;
  name: string;
  symbols: string[];
  weights: number[];
  initialValue: number;
}

export default function RealTimeMonitor() {
  const {
    isConnected,
    portfolios,
    marketData,
    riskAlerts,
    addPortfolio,
    removePortfolio,
    clearRiskAlerts,
    connect,
    disconnect
  } = useWebSocket();

  const [isMonitoring, setIsMonitoring] = useState(false);
  const [showAlerts, setShowAlerts] = useState(true);
  const [portfolioConfig, setPortfolioConfig] = useState<MonitoredPortfolio>({
    id: 'demo-portfolio-001',
    name: 'Demo Portfolio',
    symbols: SAMPLE_SYMBOLS.slice(0, 4),
    weights: [0.3, 0.25, 0.25, 0.2],
    initialValue: 25000
  });

  // Auto-connect on mount
  useEffect(() => {
    connect().catch(console.error);
    return () => {
      if (isMonitoring) {
        handleStopMonitoring();
      }
    };
  }, []);

  const handleStartMonitoring = async () => {
    try {
      if (!isConnected) {
        await connect();
      }
      
      addPortfolio(
        portfolioConfig.id,
        portfolioConfig.symbols,
        portfolioConfig.weights,
        portfolioConfig.initialValue
      );
      
      setIsMonitoring(true);
    } catch (error) {
      console.error('Failed to start monitoring:', error);
    }
  };

  const handleStopMonitoring = () => {
    if (portfolioConfig.id) {
      removePortfolio(portfolioConfig.id);
    }
    setIsMonitoring(false);
  };

  const handleToggleConnection = async () => {
    if (isConnected) {
      disconnect();
      setIsMonitoring(false);
    } else {
      await connect();
    }
  };

  const currentPortfolio = portfolios.get(portfolioConfig.id);
  const recentAlerts = riskAlerts.slice(-5).reverse();

  // Calculate portfolio performance
  const calculatePerformance = () => {
    if (!currentPortfolio) return null;
    
    const currentValue = currentPortfolio.value;
    const initialValue = portfolioConfig.initialValue;
    const change = currentValue - initialValue;
    const changePercent = (change / initialValue) * 100;
    
    return {
      currentValue,
      initialValue,
      change,
      changePercent,
      isPositive: change >= 0
    };
  };

  const performance = calculatePerformance();

  return (
    <div className="space-y-6">
      {/* Connection Status & Controls */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow border border-gray-200 dark:border-gray-700 p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">Real-Time Portfolio Monitor</h2>
          <div className="flex items-center space-x-2">
            <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${
              isConnected 
                ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-200' 
                : 'bg-red-100 dark:bg-red-900/30 text-red-800 dark:text-red-200'
            }`}>
              {isConnected ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          {/* Portfolio Configuration */}
          <div>
            <h3 className="font-medium text-gray-700 mb-2">Portfolio Configuration</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">Name:</span>
                <span className="font-medium">{portfolioConfig.name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Assets:</span>
                <span className="font-medium">{portfolioConfig.symbols.join(', ')}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Initial Value:</span>
                <span className="font-medium">${portfolioConfig.initialValue.toLocaleString()}</span>
              </div>
            </div>
          </div>

          {/* Control Buttons */}
          <div className="flex flex-col space-y-2">
            <button
              onClick={handleToggleConnection}
              className={`flex items-center justify-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                isConnected
                  ? 'bg-red-100 text-red-700 hover:bg-red-200'
                  : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
              }`}
            >
              <RefreshCw className="w-4 h-4" />
              <span>{isConnected ? 'Disconnect' : 'Connect'}</span>
            </button>

            {isConnected && (
              <button
                onClick={isMonitoring ? handleStopMonitoring : handleStartMonitoring}
                className={`flex items-center justify-center space-x-2 px-4 py-2 rounded-lg font-medium transition-colors ${
                  isMonitoring
                    ? 'bg-orange-100 text-orange-700 hover:bg-orange-200'
                    : 'bg-green-100 text-green-700 hover:bg-green-200'
                }`}
              >
                {isMonitoring ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                <span>{isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}</span>
              </button>
            )}
          </div>
        </div>
      </div>

      {/* Portfolio Performance */}
      {currentPortfolio && performance && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="font-medium text-gray-700 mb-4">Portfolio Performance</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                ${performance.currentValue.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Current Value</div>
            </div>
            
            <div className="text-center">
              <div className={`text-2xl font-bold ${performance.isPositive ? 'text-green-600' : 'text-red-600'}`}>
                {performance.isPositive ? '+' : ''}${performance.change.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Change ($)</div>
            </div>
            
            <div className="text-center">
              <div className={`text-2xl font-bold flex items-center justify-center ${
                performance.isPositive ? 'text-green-600' : 'text-red-600'
              }`}>
                {performance.isPositive ? <TrendingUp className="w-5 h-5 mr-1" /> : <TrendingDown className="w-5 h-5 mr-1" />}
                {performance.changePercent.toFixed(2)}%
              </div>
              <div className="text-sm text-gray-600">Change (%)</div>
            </div>
            
            <div className="text-center">
              <div className="text-2xl font-bold text-gray-900">
                {new Date(currentPortfolio.last_update).toLocaleTimeString()}
              </div>
              <div className="text-sm text-gray-600">Last Update</div>
            </div>
          </div>
        </div>
      )}

      {/* Live Market Data */}
      {marketData.size > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <h3 className="font-medium text-gray-700 mb-4">Live Market Data</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {Array.from(marketData.entries()).map(([symbol, data]) => {
              const isPositive = (data.change_percent || 0) >= 0;
              return (
                <div key={symbol} className="border rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-medium text-gray-900">{symbol}</span>
                    <div className={`flex items-center text-sm ${
                      isPositive ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {isPositive ? <TrendingUp className="w-3 h-3 mr-1" /> : <TrendingDown className="w-3 h-3 mr-1" />}
                      {data.change_percent ? `${data.change_percent.toFixed(2)}%` : 'N/A'}
                    </div>
                  </div>
                  <div className="text-lg font-bold text-gray-900">
                    ${data.price.toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {new Date(data.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Risk Alerts */}
      {recentAlerts.length > 0 && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-medium text-gray-700">Risk Alerts</h3>
            <div className="flex items-center space-x-2">
              <button
                onClick={() => setShowAlerts(!showAlerts)}
                className="p-1 text-gray-400 hover:text-gray-600"
              >
                {showAlerts ? <BellOff className="w-4 h-4" /> : <Bell className="w-4 h-4" />}
              </button>
              <button
                onClick={clearRiskAlerts}
                className="p-1 text-gray-400 hover:text-gray-600"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {showAlerts && (
            <div className="space-y-2">
              {recentAlerts.map((alert, index) => {
                if (alert.type !== 'risk_alert') return null;
                
                const severityColors = {
                  low: 'bg-yellow-50 border-yellow-200 text-yellow-800',
                  medium: 'bg-orange-50 border-orange-200 text-orange-800',
                  high: 'bg-red-50 border-red-200 text-red-800'
                };

                return (
                  <div key={index} className={`border rounded-lg p-3 ${severityColors[alert.severity as keyof typeof severityColors]}`}>
                    <div className="flex items-center space-x-2 mb-1">
                      <AlertTriangle className="w-4 h-4" />
                      <span className="font-medium capitalize">{alert.alert_type} Alert</span>
                      <span className="text-xs">{alert.severity.toUpperCase()}</span>
                    </div>
                    <p className="text-sm">{alert.message}</p>
                    <div className="text-xs mt-1 opacity-75">
                      {new Date(alert.timestamp).toLocaleString()}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Getting Started Message */}
      {!isMonitoring && isConnected && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="w-5 h-5 text-blue-600" />
            <h3 className="font-medium text-blue-900">Ready to Monitor</h3>
          </div>
          <p className="text-blue-800 text-sm">
            Click "Start Monitoring" to begin real-time portfolio tracking with live market data and risk alerts.
          </p>
        </div>
      )}
    </div>
  );
}