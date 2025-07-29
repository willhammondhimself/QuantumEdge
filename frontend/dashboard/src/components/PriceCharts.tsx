'use client';

import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { Calendar, TrendingUp, BarChart3 } from 'lucide-react';

import { getHistoricalData } from '@/services/api';

interface PriceChartsProps {
  symbols: string[];
}

const timeRanges = [
  { id: '1M', label: '1 Month', days: 30 },
  { id: '3M', label: '3 Months', days: 90 },
  { id: '6M', label: '6 Months', days: 180 },
  { id: '1Y', label: '1 Year', days: 365 },
];

const chartTypes = [
  { id: 'price', label: 'Price', icon: TrendingUp },
  { id: 'returns', label: 'Returns', icon: BarChart3 },
];

export default function PriceCharts({ symbols }: PriceChartsProps) {
  const [selectedRange, setSelectedRange] = useState('3M');
  const [selectedChart, setSelectedChart] = useState('price');
  const [selectedSymbols, setSelectedSymbols] = useState(symbols.slice(0, 3));

  // Calculate date range
  const endDate = new Date();
  const startDate = new Date();
  const range = timeRanges.find(r => r.id === selectedRange);
  startDate.setDate(endDate.getDate() - (range?.days || 90));

  // Fetch historical data for selected symbols
  const { data: historicalData, isLoading } = useQuery({
    queryKey: ['historical-data', selectedSymbols, selectedRange],
    queryFn: async () => {
      const promises = selectedSymbols.map(symbol => 
        getHistoricalData(
          symbol,
          startDate.toISOString().split('T')[0],
          endDate.toISOString().split('T')[0],
          '1d'
        )
      );
      
      const results = await Promise.allSettled(promises);
      const data: Record<string, any> = {};
      
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          data[selectedSymbols[index]] = result.value;
        }
      });
      
      return data;
    },
    enabled: selectedSymbols.length > 0,
  });

  // Process data for charts
  const processChartData = () => {
    if (!historicalData) return [];

    // Get all unique dates
    const allDates = new Set<string>();
    Object.values(historicalData).forEach((data: any) => {
      data?.data?.forEach((price: any) => {
        allDates.add(price.timestamp.split('T')[0]);
      });
    });

    const sortedDates = Array.from(allDates).sort();

    return sortedDates.map(date => {
      const dataPoint: any = { date };

      selectedSymbols.forEach(symbol => {
        const symbolData = historicalData[symbol];
        if (symbolData?.data) {
          const priceData = symbolData.data.find((p: any) => 
            p.timestamp.split('T')[0] === date
          );
          
          if (priceData) {
            if (selectedChart === 'price') {
              dataPoint[symbol] = priceData.close;
            } else if (selectedChart === 'returns') {
              // Calculate daily return (simplified)
              const prevPrice = symbolData.data.find((p: any, i: number) => {
                const prevDate = new Date(date);
                prevDate.setDate(prevDate.getDate() - 1);
                return p.timestamp.split('T')[0] === prevDate.toISOString().split('T')[0];
              });
              
              if (prevPrice) {
                dataPoint[symbol] = ((priceData.close - prevPrice.close) / prevPrice.close) * 100;
              }
            }
          }
        }
      });

      return dataPoint;
    });
  };

  const chartData = processChartData();
  const colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899'];

  const toggleSymbol = (symbol: string) => {
    if (selectedSymbols.includes(symbol)) {
      setSelectedSymbols(selectedSymbols.filter(s => s !== symbol));
    } else if (selectedSymbols.length < 4) {
      setSelectedSymbols([...selectedSymbols, symbol]);
    }
  };

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          {/* Symbol Selection */}
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">Symbols (max 4)</h3>
            <div className="flex flex-wrap gap-2">
              {symbols.map(symbol => (
                <button
                  key={symbol}
                  onClick={() => toggleSymbol(symbol)}
                  className={`px-3 py-1 text-sm rounded-full border transition-colors ${
                    selectedSymbols.includes(symbol)
                      ? 'bg-blue-100 border-blue-500 text-blue-700'
                      : 'bg-gray-100 border-gray-300 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {symbol}
                </button>
              ))}
            </div>
          </div>

          {/* Chart Type */}
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">Chart Type</h3>
            <div className="flex space-x-2">
              {chartTypes.map(type => {
                const Icon = type.icon;
                return (
                  <button
                    key={type.id}
                    onClick={() => setSelectedChart(type.id)}
                    className={`flex items-center space-x-2 px-3 py-2 text-sm rounded-lg transition-colors ${
                      selectedChart === type.id
                        ? 'bg-blue-100 text-blue-700'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{type.label}</span>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Time Range */}
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-2">Time Range</h3>
            <div className="flex space-x-2">
              {timeRanges.map(range => (
                <button
                  key={range.id}
                  onClick={() => setSelectedRange(range.id)}
                  className={`px-3 py-2 text-sm rounded-lg transition-colors ${
                    selectedRange === range.id
                      ? 'bg-blue-100 text-blue-700'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {range.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="bg-white rounded-lg shadow p-6">
        <div className="flex items-center space-x-2 mb-4">
          <BarChart3 className="w-5 h-5 text-gray-600" />
          <h2 className="text-lg font-semibold text-gray-900">
            {selectedChart === 'price' ? 'Price Chart' : 'Daily Returns'}
          </h2>
        </div>

        {isLoading ? (
          <div className="h-96 flex items-center justify-center">
            <div className="text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
              <p className="text-gray-600">Loading chart data...</p>
            </div>
          </div>
        ) : chartData.length > 0 ? (
          <div className="h-96">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="date" 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  tickFormatter={(value) => 
                    selectedChart === 'price' 
                      ? `$${value.toFixed(0)}` 
                      : `${value.toFixed(1)}%`
                  }
                />
                <Tooltip 
                  labelFormatter={(value) => new Date(value).toLocaleDateString()}
                  formatter={(value: any, name) => [
                    selectedChart === 'price' 
                      ? `$${value?.toFixed(2)}` 
                      : `${value?.toFixed(2)}%`,
                    name
                  ]}
                />
                <Legend />
                {selectedSymbols.map((symbol, index) => (
                  <Line
                    key={symbol}
                    type="monotone"
                    dataKey={symbol}
                    stroke={colors[index]}
                    strokeWidth={2}
                    dot={false}
                    connectNulls={false}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-96 flex items-center justify-center">
            <div className="text-center">
              <Calendar className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-600">No data available for selected symbols</p>
              <p className="text-gray-500 text-sm">Try selecting different symbols or time range</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}