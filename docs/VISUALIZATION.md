# Visualization Dashboard

Interactive real-time options analytics dashboard built with Dash and Plotly.

## Overview

The visualization dashboard provides a professional, real-time interface for analyzing option prices, volatility surfaces, and model calibrations. It reads data from the CalibrationService output and displays interactive charts.

### Features

- **Real-time Updates**: Automatic refresh (0.5s to 5s intervals)
- **Volatility Smile**: Call and put implied volatility curves
- **3D IV Surface**: Interactive 3D volatility surface
- **Model Comparison**: Model vs market price differences
- **Greeks Heatmap**: Delta, gamma, vega, theta, rho visualization
- **Model Performance**: Calibration quality metrics
- **Term Structure**: ATM IV across expiries

---

## Quick Start

### Start the Dashboard

```bash
# Make sure calibration service is running
python python/option_pricer/calibration.py

# In a new terminal, start dashboard
python python/option_pricer/visualization.py
```

Open your browser to **http://localhost:8050**

---

## Dashboard Layout

```
┌─────────────────────────────────────────────────────┐
│  Options Analytics Platform                   LIVE  │
├─────────────────────────────────────────────────────┤
│  Symbol: [AAPL▼]  Expiry: [2024-01-19▼]  Refresh:▼ │
├─────────────────────────────────────────────────────┤
│  Spot: $175.50  ATM IV: 23.5%  SABR α: 0.295  ...  │
├──────────────────────┬──────────────────────────────┤
│  Volatility Smile    │  3D IV Surface               │
│  [Chart]             │  [3D Chart]                  │
├──────────────────────┼──────────────────────────────┤
│  Model vs Market     │  Greeks Heatmap              │
│  [Chart]             │  [Heatmap]                   │
├──────────────────────┼──────────────────────────────┤
│  Model Performance   │  Term Structure              │
│  [Gauge]             │  [Chart]                     │
└──────────────────────┴──────────────────────────────┘
```

---

## Control Bar

### Symbol Dropdown

Select the underlying symbol to analyze:
- All symbols from CalibrationService
- Auto-populated from calibrations.json
- Updates expiry dropdown when changed

### Expiry Dropdown

Select the option expiry date:
- All available expiries for selected symbol
- Sorted chronologically
- Nearest expiry selected by default

### Refresh Rate

Control update frequency:
- **Real-time (0.5s)**: For active monitoring
- **Fast (1s)**: Default, good balance
- **Normal (2s)**: Reduce CPU usage
- **Slow (5s)**: Minimal updates

---

## Metrics Bar

Displays key metrics updated in real-time:

- **Spot Price**: Current underlying price
- **ATM IV**: At-the-money implied volatility
- **IV Skew**: Put-call volatility difference (90% vs 110% strikes)
- **SABR α, ρ, ν**: Calibrated SABR parameters
- **Last Update**: Timestamp of latest data

---

## Charts

### 1. Volatility Smile

**Description:** Displays implied volatility vs strike price

**Features:**
- Blue line: Call options IV
- Purple line: Put options IV
- Orange dashed line: Current spot price
- Markers show individual data points
- Shaded area under call IV curve

**Interpretation:**
- Smile shape indicates market sentiment
- Higher put IVs → Fear of downside
- Symmetric smile → Balanced risk perception

---

### 2. 3D IV Surface

**Description:** Three-dimensional volatility surface across strikes and time

**Features:**
- X-axis: Strike prices
- Y-axis: Time to expiration
- Z-axis: Implied volatility (%)
- Interactive rotation and zoom
- Viridis color scale

**Interpretation:**
- Visualize term structure of volatility
- Identify patterns across strikes and time
- Compare near-term vs far-term behavior

**Note:** Currently simulates time dimension from single expiry data

---

### 3. Model vs Market Prices

**Description:** Bar chart showing pricing errors (model - market)

**Features:**
- Green bars: Model overpriced (positive error)
- Red bars: Model underpriced (negative error)
- Shows first 10 strikes
- Based on call options

**Interpretation:**
- Small errors → Good calibration
- Large errors → Model misfit or illiquid strikes
- Systematic bias → Model issue

---

### 4. Greeks Heatmap

**Description:** Color-coded matrix of option Greeks across strikes

**Features:**
- Rows: Delta, Gamma, Vega, Theta, Rho
- Columns: Strike prices (first 15)
- Blue scale: Darker = higher values
- Values displayed in cells

**Interpretation:**
- Delta: Hedge ratios
- Gamma: Curvature exposure
- Vega: Vol sensitivity
- Theta: Time decay
- Rho: Rate sensitivity

**Note:** Greeks are approximated based on moneyness

---

### 5. Model Performance

**Description:** Gauge showing calibration quality score

**Features:**
- Score: 0-100 (higher is better)
- Based on RMSE of model vs market prices
- Color zones:
  - Red (0-50): Poor fit
  - Orange (50-85): Acceptable
  - Green (85-100): Excellent

**Calculation:**
```python
RMSE = sqrt(mean((model_price - market_price)^2))
Score = max(0, 100 - RMSE/5 * 100)
```

---

### 6. Term Structure

**Description:** ATM implied volatility across all available expiries

**Features:**
- Green line with markers: ATM IV for each expiry
- Orange star: Currently selected expiry
- X-axis: Expiry dates
- Y-axis: ATM IV (%)

**Interpretation:**
- Upward slope → Volatility increases with time (normal)
- Downward slope → Near-term volatility elevated (stress)
- Flat → Stable volatility expectations

---

## Data Source

Dashboard reads from:
```
./calibration_data_realtime/calibrations.json
```

**Required Structure:**
```json
{
  "updated": "2024-10-30T14:30:00",
  "data": {
    "SYMBOL": {
      "EXPIRY": {
        "spot": 175.50,
        "timestamp": "...",
        "models": {
          "SABR": {
            "params": {...},
            "prices": {...}
          }
        }
      }
    }
  }
}
```

---

## Configuration

### Change Port

Edit `visualization.py`:

```python
# At bottom of file
if __name__ == '__main__':
    app.run(debug=True, port=8051, host='0.0.0.0')
```

### Change Theme

Modify `styles` dictionary in `visualization.py`:

```python
styles = {
    'main_container': {
        'background': '#0a0b0d',  # Dark background
        ...
    }
}
```

### Custom Data Path

Change `possible_paths` in `read_calibration_data()`:

```python
possible_paths = [
    './my_custom_path/calibrations.json',
    ...
]
```

---

## Keyboard Shortcuts

When dashboard is focused in browser:

- **Ctrl+R**: Refresh page
- **F11**: Fullscreen
- **Ctrl++/Ctrl+-**: Zoom in/out

---

## Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with multiple workers
gunicorn -w 4 -b 0.0.0.0:8050 python.option_pricer.visualization:server
```

### Using Waitress (Windows-compatible)

```bash
pip install waitress

# Run server
waitress-serve --port=8050 python.option_pricer.visualization:server
```

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8050

CMD ["python", "python/option_pricer/visualization.py"]
```

Build and run:

```bash
docker build -t option-dashboard .
docker run -p 8050:8050 -v $(pwd)/calibration_data_realtime:/app/calibration_data_realtime option-dashboard
```

---

## Troubleshooting

### Dashboard won't start

**Solution:**
1. Check port 8050 is not in use:
   ```bash
   lsof -i :8050  # Mac/Linux
   netstat -ano | findstr :8050  # Windows
   ```
2. Try different port (see [Configuration](#configuration))

### No data displayed

**Solution:**
1. Verify calibrations.json exists:
   ```bash
   ls -l calibration_data_realtime/calibrations.json
   ```
2. Check file is not empty:
   ```bash
   cat calibration_data_realtime/calibrations.json
   ```
3. Ensure CalibrationService is running

### Charts not updating

**Solution:**
1. Check browser console for errors (F12)
2. Verify refresh interval is set
3. Restart dashboard

### Slow performance

**Solution:**
1. Increase refresh interval (2s or 5s)
2. Close other browser tabs
3. Reduce number of symbols in calibration
4. Use production WSGI server (Gunicorn)

---

## Advanced Usage

### Add Custom Chart

Edit `visualization.py`:

```python
# Add chart to layout
html.Div(style=styles['chart_card'], children=[
    html.Div('My Custom Chart', style=styles['chart_header']),
    html.Div(style=styles['chart_body'], children=[
        dcc.Graph(id='custom-chart', config={'displayModeBar': False})
    ])
])

# Add callback
@app.callback(
    Output('custom-chart', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_custom_chart(n, symbol, expiry):
    data = process_calibration_data(symbol, expiry)
    # Create your figure
    fig = go.Figure(...)
    return fig
```

### Export Charts

Use Plotly's built-in export:

```python
# Add to chart config
dcc.Graph(
    id='volatility-smile',
    config={
        'displayModeBar': True,
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'volatility_smile',
            'height': 800,
            'width': 1200,
            'scale': 2
        }
    }
)
```

Users can click camera icon to download charts.

---

## Complete Example Workflow

```bash
# Terminal 1: Start market data service
cd option_pricer
source venv/bin/activate
python -c "
from python.option_pricer.market_data import MarketDataService
service = MarketDataService(['SPY', 'AAPL', 'MSFT'])
service.start()
import time
while True:
    time.sleep(60)
"

# Terminal 2: Start calibration service
python python/option_pricer/calibration.py

# Terminal 3: Start dashboard
python python/option_pricer/visualization.py

# Open browser to http://localhost:8050
```

---

## See Also

- [Calibration Service](CALIBRATION.md)
- [Market Data Service](MARKET_DATA.md)
- [API Reference](API_REFERENCE.md)
