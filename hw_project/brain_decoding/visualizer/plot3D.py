import plotly.io as plt_io
import plotly.graph_objects as go
import numpy as np



def plot_3d(component1: np.ndarray,
            component2 : np.ndarray,
            component3 :np.ndarray,
            path:str,
            y = None,
            title: str = None) -> None:
    
    # create a list of unique categories in y
    unique_categories = np.unique(y)

    # create a list of traces for each category
    traces = []
    for i, category in enumerate(unique_categories):
        trace = go.Scatter3d(
            x=component1[y==category],
            y=component2[y==category],
            z=component3[y==category],
            mode='markers',
            name=str(category),
            marker=dict(
                size=3,
                color=y[y==category],
                colorscale='Rainbow',
                cmin=0,
                cmax=len(unique_categories) - 1,
                opacity=1,
                line_width=1
            )
        )
        traces.append(trace)

    # create the plot with multiple traces
    fig = go.Figure(data=traces)

    # set the layout with legend
    fig.update_layout(
        margin=dict(l=50,r=50,b=50,t=50), 
        width=800, 
        height=600, 
        autosize=True,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="top", 
            y=0, 
            xanchor="center", 
            x=0.5
        ),
        title = title 
    )

    # set the template
    fig.layout.template = 'plotly_dark'

    #fig.write_image(path)
    # show the plot
    fig.show()

    