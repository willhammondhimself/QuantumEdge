/**
 * API types for QuantumEdge dashboard
 */

export interface Asset {
  symbol: string;
  name: string;
  asset_type: string;
  exchange?: string;
  currency: string;
  sector?: string;
  industry?: string;
  description?: string;
  market_cap?: number;
}

export interface Price {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  adjusted_close?: number;
}

export interface HistoricalData {
  symbol: string;
  frequency: string;
  start_date: string;
  end_date: string;
  source: string;
  data: Price[];
  count: number;
}

export interface MarketMetrics {
  timestamp: string;
  vix?: number;
  spy_return?: number;
  bond_yield_10y?: number;
  dxy?: number;
  gold_price?: number;
  oil_price?: number;
}

export interface PortfolioResult {
  weights: number[];
  expected_return: number;
  expected_variance: number;
  volatility: number;
  sharpe_ratio: number;
  objective_value: number;
}

export interface OptimizationResponse {
  optimization_id: string;
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED';
  optimization_type: 'MEAN_VARIANCE' | 'ROBUST' | 'VQE' | 'QAOA';
  solve_time: number;
  success: boolean;
  message?: string;
  portfolio?: PortfolioResult;
}

export interface MeanVarianceRequest {
  expected_returns: number[];
  covariance_matrix: number[][];
  objective?: 'maximize_sharpe' | 'minimize_variance' | 'maximize_return' | 'maximize_utility' | 'minimize_cvar' | 'maximize_calmar' | 'maximize_sortino';
  risk_aversion?: number;
  risk_free_rate?: number;
  constraints?: any;
  
  // Advanced objective parameters
  cvar_confidence?: number;
  returns_data?: number[][];
  lookback_periods?: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  timestamp: string;
  services: Record<string, string>;
  active_optimizations: number;
}

// Backtesting types
export type BacktestStrategyType = 'buy_and_hold' | 'rebalancing' | 'mean_variance' | 'vqe' | 'qaoa';
export type RebalanceFrequencyType = 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annually';

export interface BacktestRequest {
  strategy_type: BacktestStrategyType;
  symbols: string[];
  start_date: string;
  end_date: string;
  initial_cash: number;
  commission_rate: number;
  min_commission: number;
  rebalance_frequency: RebalanceFrequencyType;
  target_weights?: Record<string, number>;
  risk_aversion?: number;
  lookback_period?: number;
  depth?: number;
  num_layers?: number;
  max_iterations?: number;
  cardinality_constraint?: number;
  min_weight: number;
  max_weight: number;
  allow_short_selling: boolean;
  benchmark_symbol?: string;
  risk_free_rate: number;
}

export interface PerformanceMetrics {
  total_return: number;
  annualized_return: number;
  cagr: number;
  volatility: number;
  annualized_volatility: number;
  max_drawdown: number;
  max_drawdown_duration: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  calmar_ratio: number;
  omega_ratio: number;
  downside_deviation: number;
  var_95: number;
  cvar_95: number;
  skewness: number;
  kurtosis: number;
  beta?: number;
  alpha?: number;
  information_ratio?: number;
  tracking_error?: number;
}

export interface BacktestSummary {
  initial_value: number;
  final_value: number;
  total_return: number;
  num_rebalances: number;
  total_commissions: number;
}

export interface BacktestResponse {
  backtest_id: string;
  success: boolean;
  execution_time: number;
  performance_metrics?: PerformanceMetrics;
  summary?: BacktestSummary;
  portfolio_values?: Array<{ date: string; value: number }>;
  portfolio_weights?: Array<{ date: string; [symbol: string]: any }>;
  benchmark_values?: Array<{ date: string; value: number }>;
  error_message?: string;
  config?: any;
}

export interface CompareStrategiesRequest {
  strategies: BacktestRequest[];
  strategy_names: string[];
}

export interface CompareStrategiesResponse {
  comparison_id: string;
  success: boolean;
  execution_time: number;
  results: Array<{
    name: string;
    success: boolean;
    performance_metrics?: PerformanceMetrics;
    summary?: BacktestSummary;
    error_message?: string;
  }>;
  performance_comparison?: Array<Record<string, any>>;
  portfolio_values?: Record<string, Array<{ date: string; value: number }>>;
  error_message?: string;
}