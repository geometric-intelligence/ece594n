# For plotting
import plotly.io as plt_io
import plotly.graph_objects as go
import numpy as np


def plot_2d(component1: np.ndarray, 
            component2: np.ndarray, 
            path: str, y=None,
            title: str = None) -> None:
    # Create a list of unique categories
    categories = np.unique(y)

    # Create a list of colors for each category

    traces = []
    for i, category in enumerate(categories):
        mask = y == category
        trace = go.Scatter(
            x=component1[mask],
            y=component2[mask],
            mode='markers',
            name=str(category),
            marker=dict(
                size=5,
                colorscale='Rainbow',
                opacity=0.7,
                line=dict(width=1)
            )
        )
        traces.append(trace)

    # Create the figure and add the traces
    fig = go.Figure(data=traces)

    # Set the layout
    fig.update_layout(
        margin=dict(l=100, r=100, b=100, t=100),
        width=800,
        height=600,
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="center",
            x=0.5
        ),
        title = title 
    )
    fig.layout.template = 'plotly_dark'

    #fig.write_image(path)
    # Show the plot
    fig.show()

    
    
    
