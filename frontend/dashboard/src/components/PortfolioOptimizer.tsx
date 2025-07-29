'use client';

import { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { 
  TrendingUp, 
  Zap, 
  Settings, 
  Play,
  Loader2,
  AlertCircle,
  CheckCircle,
  PieChart
} from 'lucide-react';

import { 
  optimizeMeanVariance, 
  optimizeVQE, 
  optimizeQAOA,
  generateSampleReturns,
  generateSampleCovariance,
  SAMPLE_SYMBOLS 
} from '@/services/api';

const optimizationTypes = [
  {
    id: 'mean-variance',
    name: 'Mean-Variance',
    description: 'Classical Markowitz optimization',
    icon: TrendingUp,
    color: 'bg-blue-500'
  },
  {
    id: 'vqe',
    name: 'VQE',
    description: 'Quantum eigenportfolio optimization',
    icon: Zap,
    color: 'bg-purple-500'
  },
  {
    id: 'qaoa',
    name: 'QAOA',
    description: 'Quantum combinatorial optimization',
    icon: Settings,
    color: 'bg-green-500'
  }
];

const objectiveTypes = [
  { id: 'maximize_sharpe', name: 'Maximize Sharpe Ratio', description: 'Risk-adjusted returns' },
  { id: 'minimize_variance', name: 'Minimize Variance', description: 'Lowest risk portfolio' },
  { id: 'maximize_return', name: 'Maximize Return', description: 'Highest expected return' },
  { id: 'maximize_utility', name: 'Maximize Utility', description: 'Mean-variance utility' },
  { id: 'minimize_cvar', name: 'Minimize CVaR', description: 'Conditional value at risk' },
  { id: 'maximize_sortino', name: 'Maximize Sortino', description: 'Downside risk-adjusted returns' },
  { id: 'maximize_calmar', name: 'Maximize Calmar', description: 'Drawdown-adjusted returns' }
];

export default function PortfolioOptimizer() {
  const [selectedType, setSelectedType] = useState('mean-variance');
  const [selectedObjective, setSelectedObjective] = useState('maximize_sharpe');
  const [numAssets, setNumAssets] = useState(5);
  const [riskAversion, setRiskAversion] = useState(1.0);
  const [cvarConfidence, setCvarConfidence] = useState(0.05);
  const [lookbackPeriods, setLookbackPeriods] = useState(252);
  const [result, setResult] = useState<any>(null);

  // Mutations for different optimization types
  const meanVarianceMutation = useMutation({
    mutationFn: optimizeMeanVariance,
    onSuccess: setResult,
    onError: (error) => console.error('Mean-variance optimization failed:', error),
  });

  const vqeMutation = useMutation({
    mutationFn: optimizeVQE,
    onSuccess: setResult,
    onError: (error) => console.error('VQE optimization failed:', error),
  });

  const qaoaMutation = useMutation({
    mutationFn: optimizeQAOA,
    onSuccess: setResult,
    onError: (error) => console.error('QAOA optimization failed:', error),
  });

  const handleOptimize = () => {
    const symbols = SAMPLE_SYMBOLS.slice(0, numAssets);
    const expectedReturns = generateSampleReturns(numAssets);
    const covarianceMatrix = generateSampleCovariance(numAssets);

    switch (selectedType) {
      case 'mean-variance':
        // Generate sample historical returns for advanced objectives
        const needsHistoricalData = ['minimize_cvar', 'maximize_sortino', 'maximize_calmar'].includes(selectedObjective);
        const historicalReturns = needsHistoricalData ? 
          Array.from({ length: 252 }, () => generateSampleReturns(numAssets)) : undefined;
        
        meanVarianceMutation.mutate({
          expected_returns: expectedReturns,
          covariance_matrix: covarianceMatrix,
          objective: selectedObjective as any,
          risk_aversion: riskAversion,
          cvar_confidence: cvarConfidence,
          returns_data: historicalReturns,
          lookback_periods: lookbackPeriods,
        });
        break;
      
      case 'vqe':
        vqeMutation.mutate({
          covariance_matrix: covarianceMatrix,
          depth: 3,
          optimizer: 'COBYLA',
          max_iterations: 100,
          num_eigenportfolios: 1,
          num_random_starts: 5,
        });
        break;
      
      case 'qaoa':
        qaoaMutation.mutate({
          expected_returns: expectedReturns,
          covariance_matrix: covarianceMatrix,
          risk_aversion: riskAversion,
          num_layers: 3,
          optimizer: 'COBYLA',
          max_iterations: 100,
          cardinality_constraint: Math.min(3, numAssets),
        });
        break;
    }
  };

  const isLoading = meanVarianceMutation.isPending || vqeMutation.isPending || qaoaMutation.isPending;
  const error = meanVarianceMutation.error || vqeMutation.error || qaoaMutation.error;

  return (
    <div className="space-y-6">
      {/* Optimization Type Selection */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Optimization Method</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {optimizationTypes.map((type) => {
            const Icon = type.icon;
            return (
              <button
                key={type.id}
                onClick={() => setSelectedType(type.id)}
                className={`p-4 rounded-lg border-2 transition-all ${
                  selectedType === type.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
              >
                <div className="flex items-center space-x-3">
                  <div className={`p-2 rounded-lg ${type.color}`}>
                    <Icon className="w-5 h-5 text-white" />
                  </div>
                  <div className="text-left">
                    <h3 className="font-medium text-gray-900">{type.name}</h3>
                    <p className="text-sm text-gray-600">{type.description}</p>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-white rounded-lg shadow p-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Configuration</h2>
        
        {/* Basic Configuration */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Number of Assets
            </label>
            <select
              value={numAssets}
              onChange={(e) => setNumAssets(Number(e.target.value))}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              {[3, 4, 5, 6, 7, 8].map((n) => (
                <option key={n} value={n}>{n} assets</option>
              ))}
            </select>
          </div>
          
          {selectedType !== 'vqe' && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Risk Aversion
              </label>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={riskAversion}
                onChange={(e) => setRiskAversion(Number(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Risk-seeking (0.1)</span>
                <span className="font-medium">{riskAversion}</span>
                <span>Risk-averse (5.0)</span>
              </div>
            </div>
          )}
        </div>

        {/* Objective Selection for Mean-Variance */}
        {selectedType === 'mean-variance' && (
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 mb-3">
              Optimization Objective
            </label>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {objectiveTypes.map((objective) => (
                <button
                  key={objective.id}
                  onClick={() => setSelectedObjective(objective.id)}
                  className={`p-3 text-left rounded-lg border-2 transition-all ${
                    selectedObjective === objective.id
                      ? 'border-blue-500 bg-blue-50'
                      : 'border-gray-200 hover:border-gray-300'
                  }`}
                >
                  <h4 className={`font-medium text-sm ${
                    selectedObjective === objective.id ? 'text-blue-900' : 'text-gray-900'
                  }`}>
                    {objective.name}
                  </h4>
                  <p className={`text-xs mt-1 ${
                    selectedObjective === objective.id ? 'text-blue-700' : 'text-gray-600'
                  }`}>
                    {objective.description}
                  </p>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Advanced Parameters for Risk-based Objectives */}
        {selectedType === 'mean-variance' && ['minimize_cvar', 'maximize_sortino', 'maximize_calmar'].includes(selectedObjective) && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6 p-4 bg-gray-50 rounded-lg">
            <h3 className="col-span-full text-sm font-medium text-gray-700 mb-2">Advanced Parameters</h3>
            
            {selectedObjective === 'minimize_cvar' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  CVaR Confidence Level ({(cvarConfidence * 100).toFixed(1)}%)
                </label>
                <input
                  type="range"
                  min="0.01"
                  max="0.1"
                  step="0.005"
                  value={cvarConfidence}
                  onChange={(e) => setCvarConfidence(Number(e.target.value))}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>1%</span>
                  <span>10%</span>
                </div>
              </div>
            )}
            
            {selectedObjective === 'maximize_calmar' && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Lookback Periods
                </label>
                <select
                  value={lookbackPeriods}
                  onChange={(e) => setLookbackPeriods(Number(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                >
                  <option value={126}>6 months (126 days)</option>
                  <option value={252}>1 year (252 days)</option>
                  <option value={504}>2 years (504 days)</option>
                  <option value={756}>3 years (756 days)</option>
                </select>
              </div>
            )}
          </div>
        )}

        <div className="mt-6">
          <button
            onClick={handleOptimize}
            disabled={isLoading}
            className="flex items-center space-x-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            {isLoading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Play className="w-5 h-5" />
            )}
            <span>{isLoading ? 'Optimizing...' : 'Optimize Portfolio'}</span>
          </button>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center space-x-2 text-red-800">
            <AlertCircle className="w-5 h-5" />
            <span className="font-medium">Optimization Failed</span>
          </div>
          <p className="text-red-700 text-sm mt-1">
            {error instanceof Error ? error.message : 'An unexpected error occurred'}
          </p>
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="bg-white rounded-lg shadow p-6">
          <div className="flex items-center space-x-2 mb-4">
            <CheckCircle className="w-5 h-5 text-green-600" />
            <h2 className="text-lg font-semibold text-gray-900">Optimization Results</h2>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Portfolio Metrics */}
            <div>
              <h3 className="font-medium text-gray-700 mb-3">Portfolio Metrics</h3>
              <div className="space-y-2">
                {result.portfolio && (
                  <>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Expected Return:</span>
                      <span className="font-medium">
                        {(result.portfolio.expected_return * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Volatility:</span>
                      <span className="font-medium">
                        {(result.portfolio.volatility * 100).toFixed(2)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Sharpe Ratio:</span>
                      <span className="font-medium">
                        {result.portfolio.sharpe_ratio?.toFixed(3) || 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Solve Time:</span>
                      <span className="font-medium">{result.solve_time?.toFixed(3)}s</span>
                    </div>
                  </>
                )}
              </div>
            </div>

            {/* Portfolio Weights */}
            <div>
              <h3 className="font-medium text-gray-700 mb-3">Asset Allocation</h3>
              {result.portfolio?.weights && (
                <div className="space-y-2">
                  {result.portfolio.weights.map((weight: number, index: number) => {
                    const symbol = SAMPLE_SYMBOLS[index];
                    const percentage = (weight * 100).toFixed(1);
                    return (
                      <div key={symbol} className="flex items-center justify-between">
                        <span className="text-gray-600">{symbol}:</span>
                        <div className="flex items-center space-x-2">
                          <div className="w-20 bg-gray-200 rounded-full h-2">
                            <div
                              className="bg-blue-600 h-2 rounded-full transition-all"
                              style={{ width: `${Math.abs(weight) * 100}%` }}
                            />
                          </div>
                          <span className="font-medium text-sm w-12 text-right">
                            {percentage}%
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}