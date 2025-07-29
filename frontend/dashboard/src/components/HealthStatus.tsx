'use client';

import { CheckCircle, AlertCircle, RefreshCw } from 'lucide-react';
import { HealthResponse } from '@/types/api';

interface HealthStatusProps {
  health?: HealthResponse;
  error?: boolean;
}

export default function HealthStatus({ health, error }: HealthStatusProps) {
  if (error) {
    return (
      <div className="flex items-center space-x-2 text-red-600">
        <AlertCircle className="w-4 h-4" />
        <span className="text-sm font-medium">Offline</span>
      </div>
    );
  }

  if (!health) {
    return (
      <div className="flex items-center space-x-2 text-gray-400">
        <RefreshCw className="w-4 h-4 animate-spin" />
        <span className="text-sm">Checking...</span>
      </div>
    );
  }

  const isHealthy = health.status === 'healthy';
  const hasIssues = Object.values(health.services).some(status => status !== 'healthy' && status !== 'available');

  return (
    <div className={`flex items-center space-x-2 ${
      isHealthy && !hasIssues ? 'text-green-600' : 'text-yellow-600'
    }`}>
      <CheckCircle className="w-4 h-4" />
      <div className="text-sm">
        <span className="font-medium">
          {isHealthy ? 'Online' : 'Issues'}
        </span>
        {health.active_optimizations > 0 && (
          <span className="ml-1 text-xs text-gray-500">
            ({health.active_optimizations} active)
          </span>
        )}
      </div>
    </div>
  );
}