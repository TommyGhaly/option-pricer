import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from scipy.interpolate import make_interp_spline
import plotly.express as px

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=[
    'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
])

# Global variable to store current data
current_full_data = None

# Function to read calibration data from JSON file
def read_calibration_data():
    """Read real-time calibration data from JSON file"""
    global current_full_data
    try:
        # Try different possible paths
        possible_paths = [
            './calibration_data_realtime/calibrations.json',
            'calibration_data_realtime/calibrations.json',
            '../calibration_data_realtime/calibrations.json',
            Path.home() / 'calibration_data_realtime' / 'calibrations.json'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    data = json.load(f)
                    current_full_data = data
                    return data

        # Don't print warnings repeatedly
        return None
    except Exception as e:
        # Silently handle errors
        return None

# Function to get available symbols and expiries
def get_available_options():
    """Get list of available symbols and their expiries"""
    data = read_calibration_data()
    if not data or 'data' not in data:
        return [], {}

    symbols = list(data['data'].keys())
    symbol_expiries = {}
    for symbol in symbols:
        symbol_expiries[symbol] = list(data['data'][symbol].keys())

    return symbols, symbol_expiries

# Function to process calibration data for a specific symbol/expiry
def process_calibration_data(symbol=None, expiry=None):
    """Process the calibration data into format needed for charts"""
    raw_data = current_full_data if current_full_data else read_calibration_data()
    if not raw_data or 'data' not in raw_data:
        return None

    # Use provided symbol/expiry or default to first available
    symbols = list(raw_data['data'].keys())
    if not symbols:
        return None

    if not symbol or symbol not in symbols:
        symbol = symbols[0]

    expiries = list(raw_data['data'][symbol].keys())
    if not expiries:
        return None

    if not expiry or expiry not in expiries:
        expiry = expiries[0]

    option_data = raw_data['data'][symbol][expiry]

    # Extract relevant data
    spot = option_data.get('spot', 0)
    timestamp = option_data.get('timestamp', datetime.now().isoformat())

    # Extract SABR parameters
    sabr_params = option_data.get('models', {}).get('SABR', {}).get('params', {})

    # Extract prices and Greeks
    prices_data = option_data.get('models', {}).get('SABR', {}).get('prices', {})

    # Build structured data for charts
    strikes = []
    call_model_prices = []
    call_market_prices = []
    put_model_prices = []
    put_market_prices = []
    call_ivs = []
    put_ivs = []

    for strike_str, price_info in prices_data.items():
        strike = float(strike_str)
        strikes.append(strike)

        # Call data
        call_data = price_info.get('call', {})
        call_model_prices.append(call_data.get('model', 0))
        call_market_prices.append(call_data.get('market', 0))
        call_ivs.append(call_data.get('iv', 0))

        # Put data
        put_data = price_info.get('put', {})
        put_model_prices.append(put_data.get('model', 0))
        put_market_prices.append(put_data.get('market', 0))
        put_ivs.append(put_data.get('iv', 0))

    # Calculate Greeks (approximations since not in the data)
    deltas = []
    gammas = []
    vegas = []
    thetas = []
    rhos = []

    for strike in strikes:
        moneyness = strike / spot if spot > 0 else 1

        # Approximate Greeks based on moneyness
        if moneyness < 1:  # ITM calls
            delta = 0.5 + 0.5 * (1 - moneyness)
        else:  # OTM calls
            delta = 0.5 * np.exp(-2 * (moneyness - 1))

        gamma = 0.01 * np.exp(-10 * (moneyness - 1)**2)
        vega = 50 * np.exp(-5 * (moneyness - 1)**2)
        theta = -10 * np.exp(-3 * (moneyness - 1)**2)
        rho = 20 * delta

        deltas.append(delta)
        gammas.append(gamma)
        vegas.append(vega)
        thetas.append(theta)
        rhos.append(rho)

    return {
        'symbol': symbol,
        'expiry': expiry,
        'spot': spot,
        'timestamp': timestamp,
        'sabr_params': sabr_params,
        'strikes': strikes,
        'call_model_prices': call_model_prices,
        'call_market_prices': call_market_prices,
        'put_model_prices': put_model_prices,
        'put_market_prices': put_market_prices,
        'call_ivs': call_ivs,
        'put_ivs': put_ivs,
        'deltas': deltas,
        'gammas': gammas,
        'vegas': vegas,
        'thetas': thetas,
        'rhos': rhos
    }

# Define styles
styles = {
    'main_container': {
        'background': '#0a0b0d',
        'minHeight': '100vh',
        'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        'color': '#ffffff',
        'margin': 0,
        'padding': 0
    },
    'header': {
        'background': 'linear-gradient(90deg, #1a1d24 0%, #2d3748 100%)',
        'padding': '20px 40px',
        'borderBottom': '1px solid rgba(255, 255, 255, 0.1)',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between'
    },
    'logo_section': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '15px'
    },
    'title': {
        'fontSize': '24px',
        'fontWeight': '600',
        'margin': 0,
        'color': '#ffffff'
    },
    'status_indicator': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'padding': '6px 12px',
        'background': 'rgba(16, 185, 129, 0.1)',
        'border': '1px solid rgba(16, 185, 129, 0.3)',
        'borderRadius': '20px',
        'fontSize': '13px',
        'color': '#10b981'
    },
    'control_bar': {
        'background': '#1a1d24',
        'padding': '15px 40px',
        'display': 'flex',
        'gap': '15px',
        'alignItems': 'center',
        'borderBottom': '1px solid rgba(255, 255, 255, 0.05)',
        'flexWrap': 'wrap'
    },
    'control_item': {
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '5px',
        'minWidth': '180px'
    },
    'control_label': {
        'fontSize': '11px',
        'color': '#9ca3af',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px',
        'fontWeight': '500'
    },
    'metrics_bar': {
        'background': '#13151a',
        'padding': '15px 40px',
        'display': 'flex',
        'gap': '30px',
        'overflowX': 'auto',
        'borderBottom': '1px solid rgba(255, 255, 255, 0.05)'
    },
    'metric_item': {
        'display': 'flex',
        'flexDirection': 'column',
        'gap': '3px',
        'minWidth': '100px'
    },
    'metric_label': {
        'fontSize': '10px',
        'color': '#6b7280',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px'
    },
    'metric_value': {
        'fontSize': '20px',
        'fontWeight': '600',
        'color': '#ffffff'
    },
    'content_area': {
        'padding': '20px',
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(600px, 1fr))',
        'gap': '20px',
        'maxWidth': '1920px',
        'margin': '0 auto'
    },
    'chart_card': {
        'background': '#1a1d24',
        'borderRadius': '8px',
        'border': '1px solid rgba(255, 255, 255, 0.05)',
        'overflow': 'hidden'
    },
    'chart_header': {
        'padding': '15px 20px',
        'borderBottom': '1px solid rgba(255, 255, 255, 0.05)',
        'fontSize': '14px',
        'fontWeight': '500',
        'color': '#e5e7eb'
    },
    'chart_body': {
        'padding': '10px'
    },
    'error_toast': {
        'position': 'fixed',
        'bottom': '20px',
        'right': '20px',
        'background': 'rgba(239, 68, 68, 0.9)',
        'color': 'white',
        'padding': '12px 20px',
        'borderRadius': '6px',
        'fontSize': '13px',
        'zIndex': 1000,
        'maxWidth': '300px',
        'backdropFilter': 'blur(10px)',
        'display': 'none'
    }
}

# Initial symbols and expiries
initial_symbols, initial_symbol_expiries = get_available_options()

# App layout
app.layout = html.Div(style=styles['main_container'], children=[
    # Store component for symbol expiries mapping
    dcc.Store(id='symbol-expiries-store', data=initial_symbol_expiries),

    # Header
    html.Div(style=styles['header'], children=[
        html.Div(style=styles['logo_section'], children=[
            html.Div("⚡", style={'fontSize': '28px'}),
            html.H1("Options Analytics Platform", style=styles['title'])
        ]),
        html.Div(id='connection-status', style=styles['status_indicator'], children=[
            html.Div("●", style={'color': '#10b981'}),
            html.Span("LIVE")
        ])
    ]),

    # Control Bar
    html.Div(style=styles['control_bar'], children=[
        html.Div(style=styles['control_item'], children=[
            html.Label('Symbol', style=styles['control_label']),
            dcc.Dropdown(
                id='symbol-dropdown',
                options=[{'label': sym, 'value': sym} for sym in initial_symbols],
                value=initial_symbols[0] if initial_symbols else None,
                style={
                    'backgroundColor': '#0a0b0d',
                    'borderColor': 'rgba(255, 255, 255, 0.1)',
                    'color': '#000'
                },
                placeholder='Select Symbol...'
            )
        ]),
        html.Div(style=styles['control_item'], children=[
            html.Label('Expiry', style=styles['control_label']),
            dcc.Dropdown(
                id='expiry-dropdown',
                options=[],
                value=None,
                style={
                    'backgroundColor': '#0a0b0d',
                    'borderColor': 'rgba(255, 255, 255, 0.1)',
                    'color': '#000'
                },
                placeholder='Select Expiry...'
            )
        ]),
        html.Div(style=styles['control_item'], children=[
            html.Label('Refresh', style=styles['control_label']),
            dcc.Dropdown(
                id='refresh-rate',
                options=[
                    {'label': 'Real-time (0.5s)', 'value': 500},
                    {'label': 'Fast (1s)', 'value': 1000},
                    {'label': 'Normal (2s)', 'value': 2000},
                    {'label': 'Slow (5s)', 'value': 5000}
                ],
                value=1000,
                style={
                    'backgroundColor': '#0a0b0d',
                    'borderColor': 'rgba(255, 255, 255, 0.1)',
                    'color': '#000',
                    'minWidth': '150px'
                }
            )
        ])
    ]),

    # Metrics Bar
    html.Div(style=styles['metrics_bar'], children=[
        html.Div(style=styles['metric_item'], children=[
            html.Div('Spot Price', style=styles['metric_label']),
            html.Div(id='spot-display', children='--', style=styles['metric_value'])
        ]),
        html.Div(style=styles['metric_item'], children=[
            html.Div('ATM IV', style=styles['metric_label']),
            html.Div(id='atm-iv-display', children='--', style={**styles['metric_value'], 'color': '#10b981'})
        ]),
        html.Div(style=styles['metric_item'], children=[
            html.Div('IV Skew', style=styles['metric_label']),
            html.Div(id='iv-skew-display', children='--', style={**styles['metric_value'], 'color': '#3b82f6'})
        ]),
        html.Div(style=styles['metric_item'], children=[
            html.Div('SABR α', style=styles['metric_label']),
            html.Div(id='alpha-display', children='--', style=styles['metric_value'])
        ]),
        html.Div(style=styles['metric_item'], children=[
            html.Div('SABR ρ', style=styles['metric_label']),
            html.Div(id='rho-display', children='--', style=styles['metric_value'])
        ]),
        html.Div(style=styles['metric_item'], children=[
            html.Div('SABR ν', style=styles['metric_label']),
            html.Div(id='nu-display', children='--', style=styles['metric_value'])
        ]),
        html.Div(style=styles['metric_item'], children=[
            html.Div('Last Update', style=styles['metric_label']),
            html.Div(id='timestamp-display', children='--', style={**styles['metric_value'], 'fontSize': '16px'})
        ])
    ]),

    # Content Area with Charts
    html.Div(style=styles['content_area'], children=[
        # Volatility Smile
        html.Div(style=styles['chart_card'], children=[
            html.Div('Volatility Smile', style=styles['chart_header']),
            html.Div(style=styles['chart_body'], children=[
                dcc.Graph(id='volatility-smile', config={'displayModeBar': False})
            ])
        ]),

        # 3D Surface
        html.Div(style=styles['chart_card'], children=[
            html.Div('Implied Volatility Surface', style=styles['chart_header']),
            html.Div(style=styles['chart_body'], children=[
                dcc.Graph(id='iv-surface-3d', config={'displayModeBar': False})
            ])
        ]),

        # Price Comparison
        html.Div(style=styles['chart_card'], children=[
            html.Div('Model vs Market Prices', style=styles['chart_header']),
            html.Div(style=styles['chart_body'], children=[
                dcc.Graph(id='price-waterfall', config={'displayModeBar': False})
            ])
        ]),

        # Greeks Heatmap
        html.Div(style=styles['chart_card'], children=[
            html.Div('Greeks Analysis', style=styles['chart_header']),
            html.Div(style=styles['chart_body'], children=[
                dcc.Graph(id='greeks-heatmap', config={'displayModeBar': False})
            ])
        ]),

        # Model Performance
        html.Div(style=styles['chart_card'], children=[
            html.Div('Model Performance', style=styles['chart_header']),
            html.Div(style=styles['chart_body'], children=[
                dcc.Graph(id='model-performance', config={'displayModeBar': False})
            ])
        ]),

        # Term Structure
        html.Div(style=styles['chart_card'], children=[
            html.Div('Term Structure', style=styles['chart_header']),
            html.Div(style=styles['chart_body'], children=[
                dcc.Graph(id='term-structure', config={'displayModeBar': False})
            ])
        ])
    ]),

    # Error Toast (hidden by default)
    html.Div(id='error-toast', style=styles['error_toast'], children=[
        "Data temporarily unavailable"
    ]),

    # Update interval
    dcc.Interval(
        id='interval-component',
        interval=1000,
        n_intervals=0
    )
])

# Dark theme template for plots
plot_template = {
    'layout': {
        'plot_bgcolor': '#0a0b0d',
        'paper_bgcolor': '#1a1d24',
        'font': {'color': '#e5e7eb', 'family': 'Inter, sans-serif'},
        'title': {'font': {'size': 16, 'color': '#ffffff'}},
        'xaxis': {
            'gridcolor': 'rgba(255, 255, 255, 0.05)',
            'zerolinecolor': 'rgba(255, 255, 255, 0.1)',
            'color': '#9ca3af'
        },
        'yaxis': {
            'gridcolor': 'rgba(255, 255, 255, 0.05)',
            'zerolinecolor': 'rgba(255, 255, 255, 0.1)',
            'color': '#9ca3af'
        },
        'margin': {'l': 50, 'r': 20, 't': 30, 'b': 50}
    }
}

# Color palette
colors = {
    'primary': '#3b82f6',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'purple': '#8b5cf6',
    'pink': '#ec4899'
}

# Callback to update expiry dropdown when symbol changes
@app.callback(
    Output('expiry-dropdown', 'options'),
    Output('expiry-dropdown', 'value'),
    Input('symbol-dropdown', 'value'),
    State('symbol-expiries-store', 'data')
)
def update_expiry_dropdown(symbol, symbol_expiries):
    if not symbol or symbol not in symbol_expiries:
        return [], None

    expiries = symbol_expiries[symbol]
    options = [{'label': exp, 'value': exp} for exp in expiries]
    value = expiries[0] if expiries else None

    return options, value

# Callback to update refresh rate
@app.callback(
    Output('interval-component', 'interval'),
    Input('refresh-rate', 'value')
)
def update_refresh_rate(rate):
    return rate

# Main callback to update metrics
@app.callback(
    [Output('spot-display', 'children'),
     Output('alpha-display', 'children'),
     Output('rho-display', 'children'),
     Output('nu-display', 'children'),
     Output('atm-iv-display', 'children'),
     Output('iv-skew-display', 'children'),
     Output('timestamp-display', 'children')],
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_metrics(n, symbol, expiry):
    try:
        data = process_calibration_data(symbol, expiry)
        if not data:
            return '--', '--', '--', '--', '--', '--', '--'

        # Calculate ATM IV
        spot = data.get('spot', 0)
        if spot > 0 and data.get('strikes') and data.get('call_ivs'):
            atm_strike_idx = np.argmin(np.abs(np.array(data['strikes']) - spot))
            atm_iv = data['call_ivs'][atm_strike_idx] if atm_strike_idx < len(data['call_ivs']) else 0

            # Calculate skew
            try:
                strike_90 = spot * 0.9
                strike_110 = spot * 1.1
                idx_90 = np.argmin(np.abs(np.array(data['strikes']) - strike_90))
                idx_110 = np.argmin(np.abs(np.array(data['strikes']) - strike_110))
                skew = (data['call_ivs'][idx_90] - data['call_ivs'][idx_110]) * 100
            except:
                skew = 0
        else:
            atm_iv = 0
            skew = 0

        spot_display = f"${data.get('spot', 0):.2f}"
        alpha = f"{data.get('sabr_params', {}).get('alpha', 0):.3f}"
        rho = f"{data.get('sabr_params', {}).get('rho', 0):.3f}"
        nu = f"{data.get('sabr_params', {}).get('nu', 0):.2f}"
        atm_iv_display = f"{atm_iv*100:.1f}%"
        iv_skew = f"{skew:.2f}%"

        try:
            ts = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
            timestamp = ts.strftime('%H:%M:%S')
        except:
            timestamp = datetime.now().strftime('%H:%M:%S')

        return spot_display, alpha, rho, nu, atm_iv_display, iv_skew, timestamp
    except:
        return '--', '--', '--', '--', '--', '--', '--'

# Volatility Smile
@app.callback(
    Output('volatility-smile', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_volatility_smile(n, symbol, expiry):
    try:
        data = process_calibration_data(symbol, expiry)
        if not data or not data.get('strikes') or len(data.get('strikes', [])) < 2:
            fig = go.Figure()
            fig.update_layout(plot_template['layout'], height=350)
            return fig

        fig = go.Figure()

        # Add call IV
        fig.add_trace(go.Scatter(
            x=data['strikes'],
            y=[iv * 100 for iv in data['call_ivs']],
            mode='lines+markers',
            name='Call IV',
            line=dict(color=colors['primary'], width=2),
            marker=dict(size=6),
            fill='tozeroy',
            fillcolor=f'rgba(59, 130, 246, 0.1)'
        ))

        # Add put IV
        fig.add_trace(go.Scatter(
            x=data['strikes'],
            y=[iv * 100 for iv in data['put_ivs']],
            mode='lines+markers',
            name='Put IV',
            line=dict(color=colors['purple'], width=2),
            marker=dict(size=6)
        ))

        # Add spot line
        if data.get('spot'):
            fig.add_vline(
                x=data['spot'],
                line_dash="dash",
                line_color=colors['warning'],
                line_width=1,
                annotation_text="Spot",
                annotation_position="top"
            )

        fig.update_layout(
            plot_template['layout'],
            xaxis_title='Strike',
            yaxis_title='IV (%)',
            height=350,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
        )

        return fig
    except:
        fig = go.Figure()
        fig.update_layout(plot_template['layout'], height=350)
        return fig

# 3D Surface
@app.callback(
    Output('iv-surface-3d', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_iv_surface_3d(n, symbol, expiry):
    try:
        data = process_calibration_data(symbol, expiry)
        if not data or not data.get('strikes') or not data.get('call_ivs'):
            fig = go.Figure()
            fig.update_layout(plot_template['layout'], height=350)
            return fig

        strikes = np.array(data['strikes'])
        time_to_exp = np.linspace(0.01, 0.25, 10)
        X, Y = np.meshgrid(strikes, time_to_exp)
        Z_calls = np.array([data['call_ivs']] * len(time_to_exp)) * 100

        for i, t in enumerate(time_to_exp):
            decay_factor = np.sqrt(0.25 / t)
            Z_calls[i] = Z_calls[i] * decay_factor

        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=Z_calls,
            colorscale='Viridis',
            opacity=0.9
        )])

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Strike', backgroundcolor='#0a0b0d'),
                yaxis=dict(title='Time', backgroundcolor='#0a0b0d'),
                zaxis=dict(title='IV (%)', backgroundcolor='#0a0b0d'),
                bgcolor='#0a0b0d'
            ),
            paper_bgcolor='#1a1d24',
            height=350,
            margin=dict(l=0, r=0, t=0, b=0)
        )

        return fig
    except:
        fig = go.Figure()
        fig.update_layout(plot_template['layout'], height=350)
        return fig

# Price Waterfall
@app.callback(
    Output('price-waterfall', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_price_waterfall(n, symbol, expiry):
    try:
        data = process_calibration_data(symbol, expiry)
        if not data:
            fig = go.Figure()
            fig.update_layout(plot_template['layout'], height=350)
            return fig

        errors = []
        strikes_display = []

        for i, (model, market) in enumerate(zip(data['call_model_prices'][:10], data['call_market_prices'][:10])):
            if market > 0:
                error = model - market
                errors.append(error)
                strikes_display.append(f"{data['strikes'][i]:.0f}")

        fig = go.Figure(data=[go.Bar(
            x=strikes_display,
            y=errors,
            marker_color=[colors['success'] if e >= 0 else colors['danger'] for e in errors]
        )])

        fig.update_layout(
            plot_template['layout'],
            xaxis_title='Strike',
            yaxis_title='Price Diff ($)',
            height=350
        )

        return fig
    except:
        fig = go.Figure()
        fig.update_layout(plot_template['layout'], height=350)
        return fig

# Greeks Heatmap
@app.callback(
    Output('greeks-heatmap', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_greeks_heatmap(n, symbol, expiry):
    try:
        data = process_calibration_data(symbol, expiry)
        if not data:
            fig = go.Figure()
            fig.update_layout(plot_template['layout'], height=350)
            return fig

        greeks_matrix = [
            data['deltas'][:15],
            [g * 100 for g in data['gammas'][:15]],
            [v / 10 for v in data['vegas'][:15]],
            [abs(t) / 10 for t in data['thetas'][:15]],
            [r / 20 for r in data['rhos'][:15]]
        ]

        greeks_names = ['Delta', 'Gamma', 'Vega', 'Theta', 'Rho']
        strikes_display = [f"{s:.0f}" for s in data['strikes'][:15]]

        fig = go.Figure(data=go.Heatmap(
            z=greeks_matrix,
            x=strikes_display,
            y=greeks_names,
            colorscale='Blues',
            text=[[f"{val:.3f}" for val in row] for row in greeks_matrix],
            texttemplate="%{text}",
            textfont={"size": 9}
        ))

        fig.update_layout(
            plot_template['layout'],
            xaxis_title='Strike',
            yaxis_title='',
            height=350
        )

        return fig
    except:
        fig = go.Figure()
        fig.update_layout(plot_template['layout'], height=350)
        return fig

# Model Performance
@app.callback(
    Output('model-performance', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_model_performance(n, symbol, expiry):
    try:
        data = process_calibration_data(symbol, expiry)
        if not data:
            fig = go.Figure()
            fig.update_layout(plot_template['layout'], height=350)
            return fig

        errors = []
        for model, market in zip(data['call_model_prices'], data['call_market_prices']):
            if market > 0:
                errors.append((model - market) ** 2)

        rmse = np.sqrt(np.mean(errors)) if errors else 0
        performance_score = max(0, 100 - (rmse / 5 * 100))

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=performance_score,
            title={'text': "Score", 'font': {'size': 14}},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': colors['primary']},
                'bgcolor': "rgba(255,255,255,0.05)",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(239, 68, 68, 0.2)'},
                    {'range': [50, 85], 'color': 'rgba(245, 158, 11, 0.2)'},
                    {'range': [85, 100], 'color': 'rgba(16, 185, 129, 0.2)'}
                ]
            }
        ))

        fig.update_layout(
            paper_bgcolor='#1a1d24',
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )

        return fig
    except:
        fig = go.Figure()
        fig.update_layout(plot_template['layout'], height=350)
        return fig

# Term Structure
@app.callback(
    Output('term-structure', 'figure'),
    [Input('interval-component', 'n_intervals'),
     Input('symbol-dropdown', 'value'),
     Input('expiry-dropdown', 'value')]
)
def update_term_structure(n, symbol, expiry):
    try:
        data = process_calibration_data(symbol, expiry)
        if not data or not current_full_data:
            fig = go.Figure()
            fig.update_layout(plot_template['layout'], height=350)
            return fig

        if symbol in current_full_data['data']:
            expiries_data = current_full_data['data'][symbol]

            expiry_dates = []
            atm_ivs = []

            for exp_date, exp_data in expiries_data.items():
                spot = exp_data.get('spot', 0)
                prices = exp_data.get('models', {}).get('SABR', {}).get('prices', {})

                min_diff = float('inf')
                atm_iv = 0

                for strike_str, price_info in prices.items():
                    strike = float(strike_str)
                    diff = abs(strike - spot)
                    if diff < min_diff:
                        min_diff = diff
                        atm_iv = price_info.get('call', {}).get('iv', 0)

                expiry_dates.append(exp_date)
                atm_ivs.append(atm_iv * 100)

            sorted_pairs = sorted(zip(expiry_dates, atm_ivs))
            expiry_dates, atm_ivs = zip(*sorted_pairs) if sorted_pairs else ([], [])

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=expiry_dates,
                y=atm_ivs,
                mode='lines+markers',
                name='ATM IV',
                line=dict(color=colors['success'], width=2),
                marker=dict(size=8)
            ))

            if expiry in expiry_dates:
                idx = expiry_dates.index(expiry)
                fig.add_trace(go.Scatter(
                    x=[expiry],
                    y=[atm_ivs[idx]],
                    mode='markers',
                    name='Selected',
                    marker=dict(size=12, color=colors['warning'], symbol='star')
                ))
        else:
            fig = go.Figure()

        fig.update_layout(
            plot_template['layout'],
            xaxis_title='Expiry',
            yaxis_title='ATM IV (%)',
            height=350,
            showlegend=True,
            legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
        )

        return fig
    except:
        fig = go.Figure()
        fig.update_layout(plot_template['layout'], height=350)
        return fig

# Run the app
if __name__ == '__main__':
    print("🚀 Starting Options Analytics Platform...")
    print("📊 Reading from: ./calibration_data_realtime/calibrations.json")
    print("📈 Open your browser to http://localhost:8050")

    if not read_calibration_data():
        print("⚠️  Warning: Could not find calibrations.json file!")
        print("   Please ensure the file exists at: ./calibration_data_realtime/calibrations.json")
    else:
        symbols, _ = get_available_options()
        print(f"✅ Found {len(symbols)} symbols: {', '.join(symbols)}")

    app.run(debug=True, port=8050, host='0.0.0.0')
