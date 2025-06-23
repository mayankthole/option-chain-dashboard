# 📊 Option Chain Dashboard

A comprehensive real-time dashboard for visualizing option chain data stored in PostgreSQL. Built with Streamlit and Plotly for interactive data analysis.

## 🚀 Features

### 📈 **Real-time Visualizations**
- **Spot Price Trend**: Time-series chart showing spot price movements
- **Open Interest Analysis**: CE vs PE OI comparison by strike price
- **Volume Analysis**: Trading volume comparison across strikes
- **Implied Volatility**: IV curves for both CE and PE options
- **Greeks Analysis**: Delta, Gamma, Theta, Vega visualization
- **Put-Call Ratio**: PCR trend over time with neutral line
- **OI Heatmap**: Visual heatmap of open interest data

### 📊 **Market Analytics**
- Current spot price and ATM strike
- Total CE/PE Open Interest and Volume
- Average Implied Volatility
- Put-Call Ratio calculation
- Real-time market statistics

### 🎛️ **Interactive Controls**
- Symbol selection (NIFTY, BANKNIFTY, etc.)
- Expiry date filtering
- Time range selection (100-2000 records)
- Chart visibility toggles
- Column selection for data tables
- Search functionality

### 📋 **Data Tables**
- Latest data summary
- Detailed data with customizable columns
- Real-time data filtering and search
- Export capabilities

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PostgreSQL database with option chain data
- `.env` file with database credentials

### Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements_dashboard.txt
   ```

2. **Run Dashboard**:
   ```bash
   python run_dashboard.py
   ```

3. **Access Dashboard**:
   Open your browser and go to: `http://localhost:8501`

## 📁 File Structure

```
├── dashboard.py              # Main dashboard application
├── run_dashboard.py          # Enhanced launcher script
├── requirements_dashboard.txt # Dashboard dependencies
├── database.py              # Database connection (from main app)
├── config.py                # Configuration (from main app)
└── DASHBOARD_README.md      # This file
```

## 🎯 Usage Guide

### 1. **Dashboard Overview**
- The dashboard automatically connects to your PostgreSQL database
- Real-time data updates every 30 seconds
- Responsive design works on desktop and mobile

### 2. **Sidebar Controls**
- **Symbol**: Select the underlying (NIFTY, BANKNIFTY, etc.)
- **Expiry**: Choose specific expiry or view all expiries
- **Time Range**: Select how many records to display
- **Chart Options**: Toggle individual charts on/off
- **Greek Selection**: Choose which Greek to display (Delta, Gamma, Theta, Vega)

### 3. **Market Analytics Section**
- **Current Spot**: Latest spot price
- **Put-Call Ratio**: Market sentiment indicator
- **Total OI**: Combined open interest for CE/PE
- **Average IV**: Mean implied volatility
- **Volume Analysis**: Trading activity metrics

### 4. **Chart Interpretations**

#### 📈 Spot Price Trend
- Shows price movement over time
- Helps identify trends and volatility
- Filled area shows price range

#### 📊 Open Interest Analysis
- Green bars: Call option OI
- Red bars: Put option OI
- Higher OI indicates more liquidity
- ATM strikes typically have highest OI

#### 📈 Volume Analysis
- Blue bars: Call option volume
- Orange bars: Put option volume
- High volume indicates active trading
- Useful for identifying momentum

#### 📊 Implied Volatility
- Green line: Call option IV
- Red line: Put option IV
- IV smile/skew patterns visible
- Higher IV = higher option premiums

#### 📊 Greeks Analysis
- **Delta**: Price sensitivity to underlying
- **Gamma**: Delta sensitivity to underlying
- **Theta**: Time decay
- **Vega**: Volatility sensitivity

#### 📊 Put-Call Ratio
- PCR > 1: Bearish sentiment
- PCR < 1: Bullish sentiment
- PCR = 1: Neutral market
- Red dashed line shows neutral level

#### 🔥 OI Heatmap
- Visual representation of OI distribution
- Darker colors = higher OI
- Easy to spot support/resistance levels

### 5. **Data Tables**
- **Latest Data**: Most recent records
- **Detailed Data**: Full dataset with filtering
- **Search**: Find specific values
- **Column Selection**: Customize displayed columns

## 🔧 Configuration

### Environment Variables
Make sure your `.env` file contains:
```env
DB_HOST=your_rds_endpoint
DB_PORT=5432
DB_NAME=your_database_name
DB_USER=your_username
DB_PASSWORD=your_password
```

### Dashboard Settings
- **Port**: 8501 (configurable in run_dashboard.py)
- **Auto-refresh**: 30 seconds
- **Cache TTL**: 30-60 seconds for different data types

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check `.env` file configuration
   - Verify RDS endpoint is accessible
   - Ensure database is running

2. **No Data Available**
   - Start the main data collection (`python main.py`)
   - Check if data exists in database
   - Verify symbol/expiry selection

3. **Charts Not Loading**
   - Check browser console for errors
   - Verify Plotly installation
   - Refresh the page

4. **Performance Issues**
   - Reduce time range selection
   - Disable unused charts
   - Check database performance

### Error Messages

- **"No data available"**: Start data collection first
- **"Database connection failed"**: Check credentials and network
- **"Package not found"**: Run `pip install -r requirements_dashboard.txt`

## 📊 Data Schema

The dashboard expects data in the following format:
```sql
-- Tables: option_chain_{symbol}.{symbol}_{expiry}
Columns:
- Symbol, expiry_date, fetch_time, timestamp
- Spot Price, ATM Strike, Strike Price
- CE OI, CE Chg in OI, CE Volume, CE IV, CE LTP
- CE Bid Qty, CE Bid, CE Ask, CE Ask Qty
- CE Delta, CE Theta, CE Gamma, CE Vega
- PE Bid Qty, PE Bid, PE Ask, PE Ask Qty
- PE LTP, PE IV, PE Volume, PE Chg in OI, PE OI
- PE Delta, PE Theta, PE Gamma, PE Vega
```

## 🔄 Real-time Updates

- **Auto-refresh**: Every 30 seconds
- **Manual refresh**: Click refresh button
- **Cache management**: Automatic cache invalidation
- **Data consistency**: Latest data always displayed

## 📱 Mobile Support

- Responsive design
- Touch-friendly controls
- Optimized for mobile browsers
- Sidebar collapses on small screens

## 🎨 Customization

### Styling
- Custom CSS in dashboard.py
- Color schemes and gradients
- Responsive layouts
- Professional appearance

### Charts
- Plotly templates
- Custom colors and markers
- Interactive tooltips
- Zoom and pan capabilities

## 🔒 Security

- Local server only (localhost)
- No external data transmission
- Database credentials in .env file
- CORS disabled for local use

## 📈 Performance Tips

1. **Optimize Queries**
   - Use appropriate LIMIT clauses
   - Filter by symbol/expiry
   - Cache frequently accessed data

2. **Chart Management**
   - Disable unused charts
   - Reduce data points for large datasets
   - Use appropriate time ranges

3. **Database Optimization**
   - Index on fetch_time and timestamp
   - Partition tables by date if needed
   - Regular maintenance

## 🤝 Support

For issues or questions:
1. Check this README
2. Review error messages
3. Verify database connectivity
4. Check data availability

## 📝 Changelog

### v2.0 (Current)
- Enhanced visualizations
- Real-time analytics
- Interactive controls
- Mobile support
- Performance optimizations

### v1.0
- Basic dashboard
- Simple charts
- Data tables

---

**Built with ❤️ using Streamlit & Plotly** 
