'use client';

import { useState } from 'react';
import { 
  Moon, 
  Sun, 
  Monitor, 
  Settings as SettingsIcon,
  Search,
  TrendingUp,
  AlertTriangle,
  Info,
  Zap,
  DollarSign
} from 'lucide-react';

import { useTheme } from '@/contexts/ThemeContext';
import { ASSET_CATEGORIES, STOCK_SYMBOLS, ETF_SYMBOLS, CRYPTO_SYMBOLS } from '@/services/api';

export default function Settings() {
  const { theme, setTheme } = useTheme();
  const [activeTab, setActiveTab] = useState('general');
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedAssets, setSelectedAssets] = useState<string[]>([]);

  const tabs = [
    { id: 'general', name: 'General', icon: SettingsIcon },
    { id: 'assets', name: 'Assets', icon: TrendingUp },
    { id: 'alerts', name: 'Alerts', icon: AlertTriangle },
  ];

  const themeOptions = [
    { id: 'light', name: 'Light', icon: Sun, description: 'Clean, bright interface' },
    { id: 'dark', name: 'Dark', icon: Moon, description: 'Easy on the eyes' },
    { id: 'system', name: 'System', icon: Monitor, description: 'Follow system preference' }
  ];

  const filterAssets = (symbols: string[], category: string) => {
    if (!searchTerm) return symbols;
    return symbols.filter(symbol => 
      symbol.toLowerCase().includes(searchTerm.toLowerCase())
    );
  };

  const toggleAssetSelection = (symbol: string) => {
    setSelectedAssets(prev => 
      prev.includes(symbol) 
        ? prev.filter(s => s !== symbol)
        : [...prev, symbol]
    );
  };

  const renderAssetCategory = (categoryKey: string) => {
    const category = ASSET_CATEGORIES[categoryKey as keyof typeof ASSET_CATEGORIES];
    const filteredSymbols = filterAssets(category.symbols, categoryKey);

    const getRiskColor = (riskLevel: string) => {
      switch (riskLevel) {
        case 'Low-Medium': return 'text-green-600 dark:text-green-400 bg-green-50 dark:bg-green-900/20';
        case 'Medium-High': return 'text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20';
        case 'Very High': return 'text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20';
        default: return 'text-gray-600 dark:text-gray-400 bg-gray-50 dark:bg-gray-800';
      }
    };

    return (
      <div key={categoryKey} className="bg-white dark:bg-gray-800 rounded-lg p-6 shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <span className="text-2xl">{category.icon}</span>
            <div>
              <h3 className="font-semibold text-gray-900 dark:text-gray-100">{category.name}</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400">{category.description}</p>
            </div>
          </div>
          <span className={`px-2 py-1 text-xs font-medium rounded-full ${getRiskColor(category.riskLevel)}`}>
            {category.riskLevel} Risk
          </span>
        </div>

        <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2">
          {filteredSymbols.map(symbol => (
            <button
              key={symbol}
              onClick={() => toggleAssetSelection(symbol)}
              className={`p-2 text-xs font-medium rounded border transition-all ${
                selectedAssets.includes(symbol)
                  ? 'bg-blue-100 dark:bg-blue-900/30 border-blue-500 text-blue-700 dark:text-blue-300'
                  : 'bg-gray-50 dark:bg-gray-700 border-gray-200 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-600'
              }`}
            >
              {symbol}
            </button>
          ))}
        </div>

        {categoryKey === 'crypto' && (
          <div className="mt-3 p-3 bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400" />
              <span className="text-sm font-medium text-amber-800 dark:text-amber-200">High Volatility Warning</span>
            </div>
            <p className="text-xs text-amber-700 dark:text-amber-300 mt-1">
              Cryptocurrencies are highly volatile and speculative. Only invest what you can afford to lose.
            </p>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">Settings</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Customize your QuantumEdge experience and portfolio preferences
        </p>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700">
        <div className="border-b border-gray-200 dark:border-gray-700">
          <nav className="flex space-x-8 px-6">
            {tabs.map(tab => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                      : 'border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 hover:border-gray-300 dark:hover:border-gray-600'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.name}</span>
                </button>
              );
            })}
          </nav>
        </div>

        <div className="p-6">
          {/* General Settings */}
          {activeTab === 'general' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">Theme Preferences</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {themeOptions.map(option => {
                    const Icon = option.icon;
                    const isSelected = theme === option.id || (option.id === 'system' && theme === 'system');
                    
                    return (
                      <button
                        key={option.id}
                        onClick={() => setTheme(option.id as 'light' | 'dark')}
                        className={`p-4 rounded-lg border-2 transition-all ${
                          isSelected
                            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
                            : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                        }`}
                      >
                        <div className="flex flex-col items-center space-y-2">
                          <Icon className={`w-6 h-6 ${
                            isSelected 
                              ? 'text-blue-600 dark:text-blue-400' 
                              : 'text-gray-600 dark:text-gray-400'
                          }`} />
                          <div className="text-center">
                            <h4 className={`font-medium ${
                              isSelected 
                                ? 'text-blue-900 dark:text-blue-100' 
                                : 'text-gray-900 dark:text-gray-100'
                            }`}>
                              {option.name}
                            </h4>
                            <p className={`text-xs ${
                              isSelected 
                                ? 'text-blue-700 dark:text-blue-300' 
                                : 'text-gray-600 dark:text-gray-400'
                            }`}>
                              {option.description}
                            </p>
                          </div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">Data & Performance</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <Zap className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-gray-100">Real-time Updates</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Live market data streaming</p>
                      </div>
                    </div>
                    <span className="px-3 py-1 bg-green-100 dark:bg-green-900/20 text-green-800 dark:text-green-200 text-sm font-medium rounded-full">
                      Active
                    </span>
                  </div>

                  <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div className="flex items-center space-x-3">
                      <DollarSign className="w-5 h-5 text-green-600 dark:text-green-400" />
                      <div>
                        <h4 className="font-medium text-gray-900 dark:text-gray-100">Data Source</h4>
                        <p className="text-sm text-gray-600 dark:text-gray-400">Yahoo Finance (Real-time)</p>
                      </div>
                    </div>
                    <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/20 text-blue-800 dark:text-blue-200 text-sm font-medium rounded-full">
                      Production
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Asset Selection */}
          {activeTab === 'assets' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">Asset Universe</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Select assets for portfolio optimization. Over 70+ stocks, ETFs, and cryptocurrencies available.
                </p>

                {/* Search */}
                <div className="relative mb-6">
                  <Search className="absolute left-3 top-3 w-4 h-4 text-gray-400" />
                  <input
                    type="text"
                    placeholder="Search assets by symbol..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Asset Categories */}
                <div className="space-y-6">
                  {Object.keys(ASSET_CATEGORIES).map(renderAssetCategory)}
                </div>

                {/* Selection Summary */}
                {selectedAssets.length > 0 && (
                  <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
                    <div className="flex items-center space-x-2 mb-2">
                      <Info className="w-4 h-4 text-blue-600 dark:text-blue-400" />
                      <span className="font-medium text-blue-900 dark:text-blue-100">
                        {selectedAssets.length} assets selected
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {selectedAssets.slice(0, 10).map(symbol => (
                        <span
                          key={symbol}
                          className="px-2 py-1 bg-blue-100 dark:bg-blue-800 text-blue-800 dark:text-blue-200 text-xs font-medium rounded"
                        >
                          {symbol}
                        </span>
                      ))}
                      {selectedAssets.length > 10 && (
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 text-xs font-medium rounded">
                          +{selectedAssets.length - 10} more
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Alert Settings */}
          {activeTab === 'alerts' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">Risk Alerts</h3>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  Configure when to receive alerts for portfolio risk events
                </p>

                <div className="space-y-4">
                  <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-gray-100">Volatility Alerts</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Alert when portfolio volatility exceeds threshold</p>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="2.0"
                      step="0.1"
                      defaultValue="1.0"
                      className="w-24"
                    />
                  </div>

                  <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-gray-100">Drawdown Alerts</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Alert when portfolio loses more than threshold</p>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="25"
                      step="1"
                      defaultValue="15"
                      className="w-24"
                    />
                  </div>

                  <div className="flex items-center justify-between p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
                    <div>
                      <h4 className="font-medium text-gray-900 dark:text-gray-100">Crypto Volatility</h4>
                      <p className="text-sm text-gray-600 dark:text-gray-400">Special alerts for cryptocurrency holdings</p>
                    </div>
                    <button className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-700 transition-colors">
                      Enabled
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}