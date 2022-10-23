import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_feat_hist(df, name_feat):
    fig = px.histogram(df, x=name_feat)
    return fig


def plot_feat_boxplot(df, name_feat):
    fig = px.box(df, y=name_feat)
    return fig


def plot_feat_line(df, name_feat):
    fig = px.line(df,
                  x='Параметр',
                  y=name_feat)

    return fig


def plot_line_diff(df, new_df, name_feat):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=new_df['Параметр'], y=new_df[name_feat],
                             mode='lines+markers',
                             name='Итоговый'))

    fig.add_trace(go.Scatter(x=df['Параметр'], y=df[name_feat],
                             mode='lines',
                             name='Исходный'))

    return fig
