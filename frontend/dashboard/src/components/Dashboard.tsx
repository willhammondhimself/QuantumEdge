'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { 
  Activity, 
  TrendingUp, 
  DollarSign, 
  Settings,
  BarChart3,
  RefreshCw,
  AlertCircle,
  CheckCircle
} from 'lucide-react';

import { getHealth, getMarketMetrics, SAMPLE_SYMBOLS } from '@/services/api';
import MarketOverview from './MarketOverview';
import PortfolioOptimizer from './PortfolioOptimizer';
import PriceCharts from './PriceCharts';
import BacktestDashboard from './BacktestDashboard';
import HealthStatus from './HealthStatus';

const navigation = [
  { name: 'Overview', id: 'overview', icon: BarChart3 },
  { name: 'Optimize', id: 'optimize', icon: TrendingUp },
  { name: 'Backtest', id: 'backtest', icon: Activity },
  { name: 'Charts', id: 'charts', icon: DollarSign },
  { name: 'Settings', id: 'settings', icon: Settings },
];

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');

  // Health check query
  const { data: health, isError: healthError } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  // Market metrics query
  const { data: marketMetrics, isLoading: metricsLoading } = useQuery({
    queryKey: ['market-metrics'],
    queryFn: getMarketMetrics,
    refetchInterval: 60000, // Refetch every minute
  });

  const renderContent = () => {
    switch (activeTab) {
      case 'overview':
        return <MarketOverview marketMetrics={marketMetrics} />;
      case 'optimize':
        return <PortfolioOptimizer />;
      case 'backtest':
        return <BacktestDashboard />;
      case 'charts':
        return <PriceCharts symbols={SAMPLE_SYMBOLS.slice(0, 4)} />;
      case 'settings':
        return (
          <div className="bg-white rounded-lg shadow p-6">
            <h2 className="text-xl font-semibold mb-4">Settings</h2>
            <p className="text-gray-600">Configuration options coming soon...</p>
          </div>
        );
      default:
        return <MarketOverview marketMetrics={marketMetrics} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-4">
            <div className="flex items-center">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">Q</span>
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900">QuantumEdge</h1>
                  <p className="text-xs text-gray-500">Portfolio Optimization</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <HealthStatus health={health} error={healthError} />
              {marketMetrics && (
                <div className="hidden md:flex items-center space-x-4 text-sm">
                  <div className="flex items-center">
                    <DollarSign className="w-4 h-4 text-green-500 mr-1" />
                    <span className="text-gray-600">VIX:</span>
                    <span className="font-medium ml-1">{marketMetrics.vix?.toFixed(2)}</span>
                  </div>
                  <div className="flex items-center">
                    <TrendingUp className="w-4 h-4 text-blue-500 mr-1" />
                    <span className="text-gray-600">SPY:</span>
                    <span className={`font-medium ml-1 ${
                      (marketMetrics.spy_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {((marketMetrics.spy_return || 0) * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex flex-col lg:flex-row gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:w-64 flex-shrink-0">
            <nav className="bg-white rounded-lg shadow p-4">
              <ul className="space-y-2">
                {navigation.map((item) => {
                  const Icon = item.icon;
                  return (
                    <li key={item.id}>
                      <button
                        onClick={() => setActiveTab(item.id)}
                        className={`w-full flex items-center px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                          activeTab === item.id
                            ? 'bg-blue-100 text-blue-700'
                            : 'text-gray-700 hover:bg-gray-100'
                        }`}
                      >
                        <Icon className="w-5 h-5 mr-3" />
                        {item.name}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </nav>
          </div>

          {/* Main Content */}
          <div className="flex-1">
            {renderContent()}
          </div>
        </div>
      </div>
    </div>
  );
}