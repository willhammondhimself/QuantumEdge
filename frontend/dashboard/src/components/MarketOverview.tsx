'use client';

import { useQuery } from '@tanstack/react-query';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Activity,
  Zap,
  Shield,
  Coins
} from 'lucide-react';

import { MarketMetrics } from '@/types/api';
import { getCurrentPrices, SAMPLE_SYMBOLS } from '@/services/api';

interface MarketOverviewProps {
  marketMetrics?: MarketMetrics;
}

export default function MarketOverview({ marketMetrics }: MarketOverviewProps) {
  // Get current prices for sample symbols
  const { data: prices, isLoading: pricesLoading } = useQuery({
    queryKey: ['current-prices', SAMPLE_SYMBOLS.slice(0, 6)],
    queryFn: () => getCurrentPrices(SAMPLE_SYMBOLS.slice(0, 6)),
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const formatPrice = (price: number) => `$${price.toFixed(2)}`;
  const formatPercentage = (value: number) => `${(value * 100).toFixed(2)}%`;

  const marketCards = [
    {
      title: 'Volatility Index',
      value: marketMetrics?.vix ? marketMetrics.vix.toFixed(2) : '--',
      icon: Activity,
      color: 'text-orange-600',
      bgColor: 'bg-orange-50',
      description: 'Market fear gauge'
    },
    {
      title: 'S&P 500 Return',
      value: marketMetrics?.spy_return ? formatPercentage(marketMetrics.spy_return) : '--',
      icon: marketMetrics?.spy_return && marketMetrics.spy_return >= 0 ? TrendingUp : TrendingDown,
      color: marketMetrics?.spy_return && marketMetrics.spy_return >= 0 ? 'text-green-600' : 'text-red-600',
      bgColor: marketMetrics?.spy_return && marketMetrics.spy_return >= 0 ? 'bg-green-50' : 'bg-red-50',
      description: 'Daily return'
    },
    {
      title: '10Y Treasury',
      value: marketMetrics?.bond_yield_10y ? `${marketMetrics.bond_yield_10y.toFixed(2)}%` : '--',
      icon: Shield,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      description: 'Risk-free rate proxy'
    },
    {
      title: 'Gold Price',
      value: marketMetrics?.gold_price ? formatPrice(marketMetrics.gold_price) : '--',
      icon: Coins,
      color: 'text-yellow-600',
      bgColor: 'bg-yellow-50',
      description: 'Safe haven asset'
    },
  ];

  return (
    <div className="space-y-6">
      {/* Market Metrics Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {marketCards.map((card) => {
          const Icon = card.icon;
          return (
            <div key={card.title} className="bg-white rounded-lg shadow p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600">{card.title}</p>
                  <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
                  <p className="text-xs text-gray-500 mt-1">{card.description}</p>
                </div>
                <div className={`p-2 rounded-lg ${card.bgColor}`}>
                  <Icon className={`w-6 h-6 ${card.color}`} />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Stock Prices Grid */}
      <div className="bg-white rounded-lg shadow">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-900">Live Prices</h2>
          <p className="text-sm text-gray-600">Real-time market data</p>
        </div>
        
        <div className="p-6">
          {pricesLoading ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Array.from({ length: 6 }).map((_, i) => (
                <div key={i} className="animate-pulse">
                  <div className="h-4 bg-gray-200 rounded w-16 mb-2"></div>
                  <div className="h-6 bg-gray-200 rounded w-20"></div>
                </div>
              ))}
            </div>
          ) : prices ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(prices).map(([symbol, price]) => (
                <div key={symbol} className="border rounded-lg p-4 hover:bg-gray-50 transition-colors">
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-900">{symbol}</span>
                    <Zap className="w-4 h-4 text-green-500" />
                  </div>
                  {price ? (
                    <div>
                      <p className="text-xl font-bold text-gray-900">
                        {formatPrice(price.close)}
                      </p>
                      <div className="text-sm text-gray-600 mt-1">
                        <span>Vol: {price.volume.toLocaleString()}</span>
                      </div>
                    </div>
                  ) : (
                    <p className="text-red-600 text-sm">No data</p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p className="text-gray-600">Unable to load price data</p>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Market Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h3 className="font-medium text-gray-700 mb-2">Risk Indicators</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">VIX Level:</span>
                <span className={`font-medium ${
                  marketMetrics?.vix ? 
                    marketMetrics.vix < 20 ? 'text-green-600' : 
                    marketMetrics.vix < 30 ? 'text-yellow-600' : 'text-red-600'
                  : 'text-gray-400'
                }`}>
                  {marketMetrics?.vix ? 
                    marketMetrics.vix < 20 ? 'Low' : 
                    marketMetrics.vix < 30 ? 'Elevated' : 'High'
                    : 'N/A'
                  }
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Market Regime:</span>
                <span className="font-medium text-blue-600">
                  {marketMetrics?.spy_return && marketMetrics.spy_return > 0.01 ? 'Bull' : 
                   marketMetrics?.spy_return && marketMetrics.spy_return < -0.01 ? 'Bear' : 'Neutral'}
                </span>
              </div>
            </div>
          </div>
          
          <div>
            <h3 className="font-medium text-gray-700 mb-2">Economic Indicators</h3>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-600">USD Strength:</span>
                <span className="font-medium text-gray-900">
                  {marketMetrics?.dxy ? marketMetrics.dxy.toFixed(2) : 'N/A'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Oil Price:</span>
                <span className="font-medium text-gray-900">
                  {marketMetrics?.oil_price ? formatPrice(marketMetrics.oil_price) : 'N/A'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}