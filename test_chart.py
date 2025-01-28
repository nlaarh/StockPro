import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Create some sample data
dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
values = np.random.randn(len(dates)).cumsum()

# Create a simple Plotly figure
fig = go.Figure()

# Add a trace
fig.add_trace(go.Scatter(
    x=dates,
    y=values,
    mode='lines',
    name='Test Data'
))

# Update layout
fig.update_layout(
    title='Test Chart',
    xaxis_title='Date',
    yaxis_title='Value'
)

# Display the chart
st.title('Test Chart Display')
st.plotly_chart(fig, use_container_width=True)
