import numpy as np
import plotly.graph_objects as go

def create_performance_metrics(data, symbol):
    """Create performance metrics visualization"""
    # Calculate returns
    data = data.copy()
    data['Daily_Returns'] = data['Close'].pct_change()
    data['Cumulative_Returns'] = (1 + data['Daily_Returns']).cumprod() - 1
    
    # Performance metrics
    total_return = data['Cumulative_Returns'].iloc[-1] * 100
    volatility = data['Daily_Returns'].std() * np.sqrt(252) * 100  # Annualized
    sharpe_ratio = (data['Daily_Returns'].mean() * 252) / (data['Daily_Returns'].std() * np.sqrt(252))
    
    max_drawdown = ((data['Close'] / data['Close'].expanding().max()) - 1).min() * 100
    
    metrics = {
        "Total Return": f"{total_return:.1f}%",
        "Volatility (Ann.)": f"{volatility:.1f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}",
        "Max Drawdown": f"{max_drawdown:.1f}%"
    }
    
    # Cumulative returns chart
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data['Cumulative_Returns'] * 100,
            mode='lines',
            name='Cumulative Returns',
            line=dict(color='#00ff88', width=2)
        )
    )
    
    fig.update_layout(
        title=f'{symbol} Cumulative Returns (%)',
        xaxis_title='Date',
        yaxis_title='Cumulative Return (%)',
        template='plotly_dark',
        height=400
    )
    
    return metrics, fig
