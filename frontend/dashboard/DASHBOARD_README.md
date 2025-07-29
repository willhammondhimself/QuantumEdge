# QuantumEdge Dashboard

Interactive dashboard for quantum-inspired portfolio optimization with real-time market data.

## Features

### 🔍 Market Overview
- **Real-time Market Metrics**: VIX, S&P 500 returns, treasury yields, commodities
- **Live Stock Prices**: Current prices for major stocks with volume data
- **Market Summary**: Risk indicators and economic indicators

### ⚡ Portfolio Optimization
- **Mean-Variance Optimization**: Classical Markowitz portfolio theory
- **VQE (Variational Quantum Eigensolver)**: Quantum eigenportfolio discovery
- **QAOA (Quantum Approximate Optimization Algorithm)**: Combinatorial portfolio selection
- **Interactive Configuration**: Adjust risk aversion, number of assets, and optimization parameters

### 📈 Price Charts
- **Historical Data Visualization**: Interactive charts with multiple time ranges
- **Multiple Asset Comparison**: Compare up to 4 assets simultaneously
- **Price vs Returns Views**: Switch between absolute prices and daily returns
- **Responsive Design**: Works on desktop and mobile devices

### 🏥 System Health
- **Real-time Health Monitoring**: API status and service availability
- **Active Optimization Tracking**: Monitor running optimization jobs
- **Error Handling**: Graceful degradation when services are unavailable

## Tech Stack

- **Frontend**: Next.js 15, React 18, TypeScript
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **Data Fetching**: TanStack Query (React Query)
- **HTTP Client**: Axios
- **Icons**: Lucide React

## Getting Started

### Prerequisites

1. **Backend API**: Ensure the QuantumEdge API is running on `http://localhost:8000`
2. **Node.js**: Version 18+ required
3. **Dependencies**: All dependencies are included in package.json

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

The dashboard will be available at `http://localhost:3000` (or another port if 3000 is occupied).

### Environment Variables

Create a `.env.local` file in the dashboard directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Usage

### Market Overview
1. Navigate to the **Overview** tab
2. View real-time market metrics and stock prices
3. Monitor market risk indicators and economic data

### Portfolio Optimization
1. Go to the **Optimize** tab
2. Select optimization method (Mean-Variance, VQE, or QAOA)
3. Configure parameters:
   - Number of assets (3-8)
   - Risk aversion level (0.1-5.0)
4. Click **Optimize Portfolio** to run optimization
5. View results including:
   - Expected return and volatility
   - Sharpe ratio
   - Asset allocation weights
   - Solve time

### Price Charts
1. Navigate to the **Charts** tab
2. Select symbols to compare (up to 4)
3. Choose chart type (Price or Returns)
4. Select time range (1M, 3M, 6M, 1Y)
5. Interactive charts with tooltips and zoom

## API Integration

The dashboard connects to the QuantumEdge API with the following endpoints:

- `GET /health` - System health check
- `GET /api/v1/market/metrics` - Market-wide metrics
- `GET /api/v1/market/price/{symbol}` - Current stock price
- `POST /api/v1/market/prices` - Multiple current prices
- `GET /api/v1/market/history/{symbol}` - Historical data
- `POST /api/v1/optimize/mean-variance` - Mean-variance optimization
- `POST /api/v1/quantum/vqe` - VQE optimization
- `POST /api/v1/quantum/qaoa` - QAOA optimization

## Development

### Project Structure

```
src/
├── app/                 # Next.js app router
├── components/          # React components
│   ├── Dashboard.tsx    # Main dashboard layout
│   ├── MarketOverview.tsx
│   ├── PortfolioOptimizer.tsx
│   ├── PriceCharts.tsx
│   └── HealthStatus.tsx
├── services/            # API services
│   └── api.ts
└── types/               # TypeScript types
    └── api.ts
```

### Key Components

- **Dashboard**: Main layout with navigation and routing
- **MarketOverview**: Real-time market data and metrics
- **PortfolioOptimizer**: Interactive optimization interface
- **PriceCharts**: Historical data visualization
- **HealthStatus**: System health indicator

### Styling

Uses Tailwind CSS for responsive, utility-first styling:
- Gray color palette for neutral elements
- Blue for primary actions and selections
- Green/red for positive/negative indicators
- Responsive design with mobile-first approach

## Performance

- **Caching**: React Query caches API responses
- **Refresh Rates**: 
  - Health check: 30 seconds
  - Market metrics: 60 seconds
  - Stock prices: 30 seconds
- **Error Handling**: Graceful fallbacks for failed API calls
- **Loading States**: Skeleton loaders and spinners

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## Contributing

1. Follow TypeScript best practices
2. Use Tailwind CSS for styling
3. Add proper error handling
4. Include loading states
5. Test on multiple screen sizes
6. Update documentation for new features