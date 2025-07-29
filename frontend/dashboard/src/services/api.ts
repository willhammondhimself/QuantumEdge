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

// Sample data for demo purposes
export const SAMPLE_SYMBOLS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX'];

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