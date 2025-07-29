'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend
} from 'recharts';
import { 
  Play, 
  Loader2, 
  TrendingUp, 
  BarChart3, 
  AlertCircle,
  CheckCircle2,
  Calendar,
  DollarSign,
  Activity,
  Target,
  Settings2
} from 'lucide-react';

import { runBacktest, compareStrategies, SAMPLE_SYMBOLS } from '@/services/api';
import { 
  BacktestRequest, 
  BacktestResponse, 
  CompareStrategiesRequest, 
  CompareStrategiesResponse,
  BacktestStrategyType,
  RebalanceFrequencyType
} from '@/types/api';

const strategyTypes = [
  {
    id: 'buy_and_hold',
    name: 'Buy & Hold',
    description: 'Simple buy and hold strategy',
    icon: Target,
    complexity: 'Simple'
  },
  {
    id: 'rebalancing',
    name: 'Rebalancing',
    description: 'Periodic rebalancing to target weights',
    icon: Activity,
    complexity: 'Simple'
  },
  {
    id: 'mean_variance',
    name: 'Mean-Variance',
    description: 'Classical portfolio optimization',
    icon: TrendingUp,
    complexity: 'Moderate'
  },
  {
    id: 'vqe',
    name: 'VQE',
    description: 'Quantum eigenportfolio optimization',
    icon: Settings2,
    complexity: 'Advanced'
  },
  {
    id: 'qaoa',
    name: 'QAOA',
    description: 'Quantum combinatorial optimization',
    icon: BarChart3,
    complexity: 'Advanced'
  }
];

const rebalanceFrequencies = [
  { id: 'monthly', name: 'Monthly', description: 'Rebalance monthly' },
  { id: 'quarterly', name: 'Quarterly', description: 'Rebalance quarterly' },
  { id: 'weekly', name: 'Weekly', description: 'Rebalance weekly' },
  { id: 'daily', name: 'Daily', description: 'Rebalance daily' }
];

export default function BacktestDashboard() {
  // Configuration state
  const [selectedStrategy, setSelectedStrategy] = useState('buy_and_hold');
  const [symbols, setSymbols] = useState(['AAPL', 'GOOGL', 'MSFT', 'AMZN']);
  const [startDate, setStartDate] = useState('2020-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [initialCash, setInitialCash] = useState(100000);
  const [rebalanceFreq, setRebalanceFreq] = useState('monthly');
  const [riskAversion, setRiskAversion] = useState(1.0);
  
  // Results state
  const [results, setResults] = useState<BacktestResponse | CompareStrategiesResponse | null>(null);
  const [showComparison, setShowComparison] = useState(false);
  
  // Backtest mutation
  const backtestMutation = useMutation<BacktestResponse, Error, BacktestRequest>({
    mutationFn: runBacktest,
    onSuccess: (data) => {
      setResults(data);
      setShowComparison(false);
    },
    onError: (error) => {
      console.error('Backtest failed:', error);
    }
  });
  
  // Strategy comparison mutation
  const comparisonMutation = useMutation<CompareStrategiesResponse, Error, CompareStrategiesRequest>({
    mutationFn: compareStrategies,
    onSuccess: (data) => {
      setResults(data);
      setShowComparison(true);
    },
    onError: (error) => {
      console.error('Comparison failed:', error);
    }
  });

  const handleRunBacktest = () => {
    const config: BacktestRequest = {
      strategy_type: selectedStrategy as BacktestStrategyType,
      symbols,
      start_date: startDate,
      end_date: endDate,
      initial_cash: initialCash,
      rebalance_frequency: rebalanceFreq as RebalanceFrequencyType,
      risk_aversion: riskAversion,
      commission_rate: 0.001,
      min_commission: 1.0,
      risk_free_rate: 0.02,
      min_weight: 0.0,
      max_weight: 1.0,
      allow_short_selling: false
    };

    backtestMutation.mutate(config);
  };

  const handleCompareStrategies = () => {
    const baseConfig = {
      symbols,
      start_date: startDate,
      end_date: endDate,
      initial_cash: initialCash,
      rebalance_frequency: rebalanceFreq as RebalanceFrequencyType,
      commission_rate: 0.001,
      min_commission: 1.0,
      risk_free_rate: 0.02,
      min_weight: 0.0,
      max_weight: 1.0,
      allow_short_selling: false
    };

    const strategies: BacktestRequest[] = [
      { ...baseConfig, strategy_type: 'buy_and_hold' },
      { ...baseConfig, strategy_type: 'mean_variance', risk_aversion: riskAversion },
      { ...baseConfig, strategy_type: 'vqe' }
    ];

    const strategyNames = ['Buy & Hold', 'Mean-Variance', 'VQE'];

    const request: CompareStrategiesRequest = {
      strategies,
      strategy_names: strategyNames
    };

    comparisonMutation.mutate(request);
  };

  const isLoading = backtestMutation.isPending || comparisonMutation.isPending;
  const error = backtestMutation.error || comparisonMutation.error;

  const formatCurrency = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value);

  const formatPercent = (value: number) => 
    new Intl.NumberFormat('en-US', { style: 'percent', minimumFractionDigits: 2 }).format(value);

  return (
    <div className="space-y-8">
      {/* Header Section - Clean and purposeful */}
      <div className="bg-white rounded-2xl border border-gray-100 p-8">
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-semibold text-gray-900 mb-2">Backtesting</h1>
            <p className="text-gray-600">Test your strategies against historical data</p>
          </div>
          <div className="flex items-center space-x-3">
            <button
              onClick={handleRunBacktest}
              disabled={isLoading}
              className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-xl hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-sm hover:shadow-md"
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              <span className="font-medium">Run Test</span>
            </button>
            <button
              onClick={handleCompareStrategies}
              disabled={isLoading}
              className="flex items-center space-x-2 px-6 py-3 border border-gray-200 text-gray-700 rounded-xl hover:bg-gray-50 transition-all duration-200"
            >
              <BarChart3 className="w-4 h-4" />
              <span className="font-medium">Compare</span>
            </button>
          </div>
        </div>

        {/* Strategy Selection - Thoughtful grid layout */}
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
          {strategyTypes.map((strategy) => {
            const Icon = strategy.icon;
            const isSelected = selectedStrategy === strategy.id;
            
            return (
              <button
                key={strategy.id}
                onClick={() => setSelectedStrategy(strategy.id)}
                className={`p-4 rounded-xl border-2 transition-all duration-200 text-left ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50 shadow-sm'
                    : 'border-gray-100 hover:border-gray-200 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-center space-x-3 mb-2">
                  <Icon className={`w-5 h-5 ${isSelected ? 'text-blue-600' : 'text-gray-500'}`} />
                  <div>
                    <h3 className={`font-medium text-sm ${isSelected ? 'text-blue-900' : 'text-gray-900'}`}>
                      {strategy.name}
                    </h3>
                    <p className={`text-xs ${isSelected ? 'text-blue-700' : 'text-gray-500'}`}>
                      {strategy.complexity}
                    </p>
                  </div>
                </div>
                <p className={`text-xs leading-relaxed ${isSelected ? 'text-blue-800' : 'text-gray-600'}`}>
                  {strategy.description}
                </p>
              </button>
            );
          })}
        </div>

        {/* Configuration Panel - Clean, organized sections */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Assets */}
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900 flex items-center space-x-2">
              <Target className="w-4 h-4 text-gray-500" />
              <span>Assets</span>
            </h3>
            <div className="space-y-3">
              <div className="flex flex-wrap gap-2">
                {SAMPLE_SYMBOLS.map(symbol => (
                  <button
                    key={symbol}
                    onClick={() => {
                      if (symbols.includes(symbol)) {
                        setSymbols(symbols.filter(s => s !== symbol));
                      } else {
                        setSymbols([...symbols, symbol]);
                      }
                    }}
                    className={`px-3 py-1.5 text-sm rounded-lg border transition-all duration-200 ${
                      symbols.includes(symbol)
                        ? 'bg-blue-100 border-blue-300 text-blue-800'
                        : 'bg-gray-50 border-gray-200 text-gray-600 hover:bg-gray-100'
                    }`}
                  >
                    {symbol}
                  </button>
                ))}
              </div>
              <p className="text-xs text-gray-500">
                Selected: {symbols.length} assets
              </p>
            </div>
          </div>

          {/* Time Period */}
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900 flex items-center space-x-2">
              <Calendar className="w-4 h-4 text-gray-500" />
              <span>Time Period</span>
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-600 mb-1">Start Date</label>
                <input
                  type="date"
                  value={startDate}
                  onChange={(e) => setStartDate(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-600 mb-1">End Date</label>
                <input
                  type="date"
                  value={endDate}
                  onChange={(e) => setEndDate(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                />
              </div>
            </div>
          </div>

          {/* Parameters */}
          <div className="space-y-4">
            <h3 className="font-medium text-gray-900 flex items-center space-x-2">
              <Settings2 className="w-4 h-4 text-gray-500" />
              <span>Parameters</span>
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-sm text-gray-600 mb-1">Initial Capital</label>
                <div className="relative">
                  <DollarSign className="w-4 h-4 text-gray-400 absolute left-3 top-2.5" />
                  <input
                    type="number"
                    value={initialCash}
                    onChange={(e) => setInitialCash(Number(e.target.value))}
                    className="w-full pl-10 pr-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm text-gray-600 mb-1">Rebalance Frequency</label>
                <select
                  value={rebalanceFreq}
                  onChange={(e) => setRebalanceFreq(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-200 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm"
                >
                  {rebalanceFrequencies.map(freq => (
                    <option key={freq.id} value={freq.id}>{freq.name}</option>
                  ))}
                </select>
              </div>

              {(selectedStrategy === 'mean_variance' || selectedStrategy === 'qaoa') && (
                <div>
                  <label className="block text-sm text-gray-600 mb-1">
                    Risk Aversion ({riskAversion.toFixed(1)})
                  </label>
                  <input
                    type="range"
                    min="0.1"
                    max="5.0"
                    step="0.1"
                    value={riskAversion}
                    onChange={(e) => setRiskAversion(Number(e.target.value))}
                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer slider"
                  />
                  <div className="flex justify-between text-xs text-gray-500 mt-1">
                    <span>Risk Seeking</span>
                    <span>Risk Averse</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4">
          <div className="flex items-center space-x-2 text-red-800">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Test Failed</span>
          </div>
          <p className="text-red-700 text-sm mt-1">
            {error instanceof Error ? error.message : 'An unexpected error occurred'}
          </p>
        </div>
      )}

      {/* Results Section - Clean, data-driven presentation */}
      {results && (
        <div className="space-y-6">
          {/* Success Indicator */}
          <div className="bg-green-50 border border-green-200 rounded-xl p-4">
            <div className="flex items-center space-x-2 text-green-800">
              <CheckCircle2 className="w-5 h-5" />
              <span className="font-medium">
                {showComparison ? 'Strategy Comparison Complete' : 'Backtest Complete'}
              </span>
            </div>
            <p className="text-green-700 text-sm mt-1">
              Execution time: {results.execution_time?.toFixed(2)}s
            </p>
          </div>

          {showComparison && 'performance_comparison' in results ? (
            // Comparison Results
            <div className="space-y-6">
              {/* Performance Comparison Table */}
              <div className="bg-white rounded-2xl border border-gray-100 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Performance Comparison</h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-gray-100">
                        <th className="text-left py-3 px-4 font-medium text-gray-600">Strategy</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-600">Total Return</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-600">CAGR</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-600">Volatility</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-600">Sharpe Ratio</th>
                        <th className="text-right py-3 px-4 font-medium text-gray-600">Max Drawdown</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.performance_comparison?.map((row: Record<string, string>, index: number) => (
                        <tr key={index} className="border-b border-gray-50 hover:bg-gray-25">
                          <td className="py-3 px-4 font-medium text-gray-900">{row.Strategy}</td>
                          <td className="py-3 px-4 text-right text-gray-700">{row['Total Return']}</td>
                          <td className="py-3 px-4 text-right text-gray-700">{row.CAGR}</td>
                          <td className="py-3 px-4 text-right text-gray-700">{row.Volatility}</td>
                          <td className="py-3 px-4 text-right text-gray-700">{row['Sharpe Ratio']}</td>
                          <td className="py-3 px-4 text-right text-gray-700">{row['Max Drawdown']}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Comparison Chart */}
              {results.portfolio_values && (
                <div className="bg-white rounded-2xl border border-gray-100 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Value Comparison</h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="date" 
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                        />
                        <YAxis 
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => formatCurrency(value)}
                        />
                        <Tooltip 
                          labelFormatter={(value) => new Date(value).toLocaleDateString()}
                          formatter={(value: number) => [formatCurrency(value), 'Portfolio Value']}
                        />
                        <Legend />
                        {Object.entries(results.portfolio_values).map(([strategy, data], index) => (
                          <Line
                            key={strategy}
                            type="monotone"
                            dataKey="value"
                            data={data}
                            stroke={['#3B82F6', '#EF4444', '#10B981'][index]}
                            strokeWidth={2}
                            dot={false}
                            name={strategy}
                          />
                        ))}
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          ) : (
            // Single Strategy Results
            <div className="space-y-6">
              {/* Key Metrics Cards */}
              {'performance_metrics' in results && results.performance_metrics && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-white rounded-xl border border-gray-100 p-4">
                    <div className="text-2xl font-bold text-gray-900">
                      {formatPercent(results.performance_metrics.total_return)}
                    </div>
                    <div className="text-sm text-gray-600">Total Return</div>
                  </div>
                  <div className="bg-white rounded-xl border border-gray-100 p-4">
                    <div className="text-2xl font-bold text-gray-900">
                      {results.performance_metrics.sharpe_ratio.toFixed(2)}
                    </div>
                    <div className="text-sm text-gray-600">Sharpe Ratio</div>
                  </div>
                  <div className="bg-white rounded-xl border border-gray-100 p-4">
                    <div className="text-2xl font-bold text-gray-900">
                      {formatPercent(results.performance_metrics.max_drawdown)}
                    </div>
                    <div className="text-sm text-gray-600">Max Drawdown</div>
                  </div>
                  <div className="bg-white rounded-xl border border-gray-100 p-4">
                    <div className="text-2xl font-bold text-gray-900">
                      {formatPercent(results.performance_metrics.annualized_volatility)}
                    </div>
                    <div className="text-sm text-gray-600">Volatility</div>
                  </div>
                </div>
              )}

              {/* Portfolio Value Chart */}
              {'portfolio_values' in results && results.portfolio_values && (
                <div className="bg-white rounded-2xl border border-gray-100 p-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Portfolio Value Over Time</h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={results.portfolio_values}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                        <XAxis 
                          dataKey="date" 
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => new Date(value).toLocaleDateString()}
                        />
                        <YAxis 
                          tick={{ fontSize: 12 }}
                          tickFormatter={(value) => formatCurrency(value)}
                        />
                        <Tooltip 
                          labelFormatter={(value) => new Date(value).toLocaleDateString()}
                          formatter={(value: number) => [formatCurrency(value), 'Portfolio Value']}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#3B82F6"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}