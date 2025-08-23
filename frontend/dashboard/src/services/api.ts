/**
 * API service functions for QuantumEdge
 */

import axios from 'axios';
import {
  Asset,
  Price,
  HistoricalData,
  MarketMetrics,
  OptimizationResponse,
  MeanVarianceRequest,
  HealthResponse,
  BacktestRequest,
  BacktestResponse,
  CompareStrategiesRequest,
  CompareStrategiesResponse
} from '@/types/api';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// Health check
export const getHealth = async (): Promise<HealthResponse> => {
  const response = await api.get('/health');
  return response.data;
};

// Market data APIs
export const getAssetInfo = async (symbol: string): Promise<Asset> => {
  const response = await api.get(`/api/v1/market/asset/${symbol}`);
  return response.data;
};

export const getCurrentPrice = async (symbol: string): Promise<Price> => {
  const response = await api.get(`/api/v1/market/price/${symbol}`);
  return response.data;
};

export const getCurrentPrices = async (symbols: string[]): Promise<Record<string, Price | null>> => {
  const response = await api.post('/api/v1/market/prices', symbols);
  return response.data;
};

export const getHistoricalData = async (
  symbol: string,
  startDate: string,
  endDate: string,
  frequency: string = '1d'
): Promise<HistoricalData> => {
  const response = await api.get(
    `/api/v1/market/history/${symbol}?start_date=${startDate}&end_date=${endDate}&frequency=${frequency}`
  );
  return response.data;
};

export const getMarketMetrics = async (): Promise<MarketMetrics> => {
  const response = await api.get('/api/v1/market/metrics');
  return response.data;
};

export const getMarketHealth = async () => {
  const response = await api.get('/api/v1/market/health');
  return response.data;
};

// Portfolio optimization APIs
export const optimizeMeanVariance = async (request: MeanVarianceRequest): Promise<OptimizationResponse> => {
  const response = await api.post('/api/v1/optimize/mean-variance', request);
  return response.data;
};

export const optimizeVQE = async (request: any): Promise<OptimizationResponse> => {
  const response = await api.post('/api/v1/quantum/vqe', request);
  return response.data;
};

export const optimizeQAOA = async (request: any): Promise<OptimizationResponse> => {
  const response = await api.post('/api/v1/quantum/qaoa', request);
  return response.data;
};

// Backtesting APIs
export const runBacktest = async (request: BacktestRequest): Promise<BacktestResponse> => {
  const response = await api.post('/api/v1/backtest/run', request);
  return response.data;
};

export const compareStrategies = async (request: CompareStrategiesRequest): Promise<CompareStrategiesResponse> => {
  const response = await api.post('/api/v1/backtest/compare', request);
  return response.data;
};

// Comprehensive asset universe for portfolio optimization
export const STOCK_SYMBOLS = [
  // Technology
  'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'CRM', 'ORCL', 'ADBE', 'INTC', 'AMD', 'ASML',
  // Finance
  'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'PYPL', 'SQ',
  // Healthcare
  'JNJ', 'PFE', 'UNH', 'MRNA', 'ABBV', 'TMO', 'DHR', 'BMY',
  // Consumer
  'WMT', 'TGT', 'COST', 'HD', 'NKE', 'SBUX', 'DIS', 'MCD',
  // Energy
  'XOM', 'CVX', 'COP', 'EOG', 'SLB'
];

export const ETF_SYMBOLS = [
  // Broad Market
  'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO',
  // Sector ETFs
  'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLP', 'XLY', 'XLU',
  // Special ETFs (User requested)
  'BITX', 'UPRO', 'ARKK', 'ARKQ', 'ARKW'
];

export const CRYPTO_SYMBOLS = [
  // Major Cryptocurrencies (User requested + others)
  'BTC-USD', 'ETH-USD', 'XRP-USD', 'DOGE-USD', 'PENGU-USD',
  // Additional Major Cryptos
  'ADA-USD', 'SOL-USD', 'MATIC-USD', 'DOT-USD', 'AVAX-USD'
];

// Asset categories for UI organization
export const ASSET_CATEGORIES = {
  stocks: {
    name: 'Stocks',
    description: 'Individual company stocks',
    symbols: STOCK_SYMBOLS,
    icon: '📈',
    riskLevel: 'Medium-High'
  },
  etfs: {
    name: 'ETFs',
    description: 'Exchange-traded funds',
    symbols: ETF_SYMBOLS,
    icon: '📊',
    riskLevel: 'Low-Medium'
  },
  crypto: {
    name: 'Cryptocurrency',
    description: 'Digital assets (High volatility)',
    symbols: CRYPTO_SYMBOLS,
    icon: '₿',
    riskLevel: 'Very High'
  }
};

// Combined symbols for backwards compatibility
export const SAMPLE_SYMBOLS = [
  ...STOCK_SYMBOLS.slice(0, 8), // Top 8 stocks
  ...ETF_SYMBOLS.slice(0, 4),   // Top 4 ETFs
  ...CRYPTO_SYMBOLS.slice(0, 3)  // Top 3 cryptos
];

// All available symbols
export const ALL_SYMBOLS = [...STOCK_SYMBOLS, ...ETF_SYMBOLS, ...CRYPTO_SYMBOLS];

export const generateSampleReturns = (numAssets: number): number[] => {
  return Array.from({ length: numAssets }, () => Math.random() * 0.2 - 0.05); // -5% to 15% returns
};

export const generateSampleCovariance = (numAssets: number): number[][] => {
  const matrix: number[][] = [];
  for (let i = 0; i < numAssets; i++) {
    matrix[i] = [];
    for (let j = 0; j < numAssets; j++) {
      if (i === j) {
        matrix[i][j] = Math.random() * 0.1 + 0.01; // Variance between 1% and 11%
      } else {
        const correlation = Math.random() * 0.6 - 0.3; // Correlation between -30% and 30%
        matrix[i][j] = correlation * Math.sqrt(matrix[i][i] || 0.05) * Math.sqrt(matrix[j][j] || 0.05);
      }
    }
  }
  return matrix;
};