#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : plot.py
# Author: anyongjin
# Date  : 2023/2/21
from pandas import DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Union
from banbot.compute.utils import logger


def _add_indicator(fig, df, inds):
    import pandas as pd
    x_labels = df['date']
    for ind in inds:
        name = ind['col']
        row_num = ind.get('row', 1)
        dtype = ind.get('type', 'scatter')
        func_args = dict(x=x_labels, y=df[name], name=name)
        if row_num == 1:
            func_args['hoverinfo'] = 'skip'
        if ind.get('color'):
            func_args['line'] = dict(color=ind.get('color'))
        if ind.get('opacity'):
            func_args['opacity'] = ind.get('opacity')
        if dtype == 'scatter':
            trace_func = go.Scatter
        elif dtype == 'mark':
            trace_func = go.Scatter
            col_type = df[name].dtype
            is_str_col = pd.api.types.is_string_dtype(col_type)
            if not is_str_col and str(col_type).find('int') < 0:
                raise ValueError('`mark` type ind must be type: str/int')
            if is_str_col:
                singels = df[df[name].notnull() & (df[name] != '')]
                func_args['mode'] = 'markers+text'
                func_args['text'] = singels[name]
                func_args['hoverinfo'] = 'text'
            else:
                singels = df[df[name] == 1]
                func_args['mode'] = 'markers'
            if not len(singels):
                continue
            func_args['x'] = singels.date
            func_args['y'] = singels.close
            func_args['marker'] = dict(
                symbol=ind['symbol'],
                size=9,
                line=dict(width=1),
                color=ind.get('color'),
            )
        elif dtype == 'line':
            end_key = ind['end_key']

            trace_func = go.Scatter
            func_args['x'] = ind.get('x')
            func_args['y'] = ind.get('y')
        elif dtype == 'bar':
            trace_func = go.Bar
        else:
            raise ValueError(f'unsupport trace type: {dtype}')
        trace_args = dict(row=row_num, col=1)
        if row_num == 1:
            trace_args['secondary_y'] = True
        fig.add_trace(trace_func(**func_args), **trace_args)


def plot_fin(org_df: DataFrame, inds: Optional[List[Union[dict, str]]] = None, row_heights: Optional[List[float]] = None,
             height=None, width=None, xaxis_width: int = 300, show_mode: str = 'inline'):
    '''
    显示蜡烛图和指标。默认绘制蜡烛图和交易量。
    :param df: 需要输出的数据
    :param inds: 指标列表[{col='volume', row=1, type='scatter|mark|bar', color='red', opacity=0.7, symbol=''}]
    :param row_heights: 多个子图的比例，需要和指标中最大row保持一致
    :param height:
    :param width:
    :param xaxis_width: 显示的蜡烛数
    :param show_mode: inline|external
    :return:
    '''
    max_rend_width = 1000
    half_rend_width = round(max_rend_width / 2)
    # 滑动选择器的数量以显示窗口的一半为步长
    win_size = round(len(org_df) / half_rend_width)
    max_rows = 1
    rg_start, rg_end = 0, max_rend_width
    df = org_df[rg_start:rg_end]
    if inds:
        inds = [dict(col=i) if isinstance(i, str) else i for i in inds]
        max_rows = max([i.get('row', 1) for i in inds])
    if not row_heights:
        row_heights = [1.] + [0.15] * (max_rows - 1)
    if not height:
        height = 600 * sum(row_heights)
    row_specs = [[{"secondary_y": True}]]
    if max_rows > 1:
        row_specs.extend([[{"secondary_y": False}]] * (max_rows - 1))
    x_labels = df['date']

    def update_yaxis(fig, xmin, xmax):
        price_max = df.iloc[xmin:xmax]['high'].max()
        price_min = df.iloc[xmin:xmax]['low'].min()
        p_delta = (price_max - price_min) * 0.1
        vol_min = df.iloc[xmin:xmax]['volume'].min()
        vol_max = df.iloc[xmin:xmax]['volume'].max()
        v_delta = (vol_max - vol_min) * 0.1
        xstart, xend = x_labels.iloc[xmin], x_labels.iloc[xmax]
        fig.update_layout(xaxis_range=(xstart, xend), yaxis2_range=[price_min - p_delta, price_max + p_delta],
                          yaxis1_range=[vol_min, vol_max + v_delta])

    def make_graph(xmin, xmax):
        fig = make_subplots(rows=max_rows, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.05,
                            specs=row_specs)
        candles = go.Candlestick(x=x_labels, open=df['open'], high=df['high'], low=df['low'],
                                 close=df['close'], name='market', showlegend=False)
        fig.add_trace(candles, secondary_y=True)
        vol_name, vol_long_name = 'volume', 'long_vol'
        vol_color, vol_long_color = "rgba(128,128,128,0.3)", "rgba(128,128,128,0.5)"
        vol_bar = go.Bar(x=x_labels, y=df[vol_name], name=vol_name, showlegend=False, marker={"color": vol_color},
                         hoverinfo='skip')
        fig.add_trace(vol_bar, secondary_y=False)
        if vol_long_name in df:
            vol_long_bar = go.Bar(x=x_labels, y=df[vol_long_name], name=vol_long_name, showlegend=False,
                                  marker={"color": vol_long_color}, hoverinfo='skip', )
            fig.add_trace(vol_long_bar, secondary_y=False)
        if inds:
            _add_indicator(fig, df, inds)
        fig_margin = dict(l=0, r=0, t=0, b=0)
        fig.update_layout(height=height, width=width, xaxis_rangeslider_visible=False,
                          hovermode='x', dragmode='pan', margin=fig_margin, barmode='overlay')
        update_yaxis(fig, xmin, xmax)
        # K线图鼠标显示竖线
        fig.update_xaxes(showspikes=True, spikethickness=1, spikedash='solid', spikemode='across', spikesnap='cursor')
        # 确保所有子图都能悬停时显示竖线
        fig.update_traces(xaxis="x1", showlegend=False)
        return fig

    # Dash 交互部分
    from jupyter_dash import JupyterDash
    from dash import html, dcc, Input, Output, State
    app = JupyterDash(__name__)
    figure = make_graph(0, xaxis_width)
    # 设置range selector的提示
    range_style = dict(appearance='none', width='100%', background='#666', height='6px', borderRadius='3px')
    app.layout = html.Div(children=[
        dcc.Graph(id='graph1', figure=figure, config=dict(displayModeBar=False)),
        dcc.Input(id='range_sel', type='range', step=1, value=0, max=win_size - 1, style=range_style)
    ])
    x_indexs = {k.strftime('%Y-%m-%d %H:%M:%S'): v for v, k in enumerate(x_labels.tolist())}
    last_range_val = 0

    def set_org_range(start, end, xmin: Optional[str] = None, xmax: Optional[str] = None):
        nonlocal x_labels, df, figure, x_indexs, rg_start, rg_end
        rg_start, rg_end = start, end
        df = org_df[start: end]
        # logger.warning(f'set org range: {xmin}  {xmax}')
        x_labels = df['date']
        x_indexs = {k.strftime('%Y-%m-%d %H:%M:%S'): v for v, k in enumerate(x_labels.tolist())}
        if not xmax:
            xmin, xmax = 0, min(xaxis_width, end - start)
        else:
            xmin, xmax = x_indexs[xmin], x_indexs[xmax]
        figure = make_graph(xmin, xmax)
        return figure

    @app.callback(
        Output('graph1', 'figure'),
        Output('range_sel', 'value'),
        Input('graph1', 'relayoutData'),
        Input('range_sel', 'value'),
        prevent_initial_call=True
    )
    def update_layout(rel_out, range_val):
        nonlocal last_range_val
        range_val = int(range_val)
        if range_val != last_range_val:
            last_range_val = range_val
            start = range_val * half_rend_width
            end = (range_val + 2) * half_rend_width
            if range_val + 2 >= win_size:
                end = len(org_df) - 1
            set_org_range(start, end)
        elif "xaxis.range[0]" in rel_out:
            xmin, xmax = rel_out["xaxis.range[0]"], rel_out["xaxis.range[1]"]
            xmin = xmin[:xmin.find('.')]
            xmax = xmax[:xmax.find('.')]
            # logger.warning(f'xaxis:  {xmin}  {xmax}')
            if xmin not in x_indexs:
                xminid, xmaxid = 0, min(len(x_indexs) - 1, xaxis_width)
                if rg_start >= half_rend_width:
                    start = max(0, rg_start - half_rend_width)
                    set_org_range(start, start + max_rend_width, xmin, xmax)
                    last_range_val = range_val = round(start / half_rend_width)
                    return figure, range_val
            elif xmax not in x_indexs:
                xminid, xmaxid = max(0, len(x_indexs) - xaxis_width), len(x_indexs) - 1
                if rg_end + half_rend_width <= len(org_df):
                    end = min(len(org_df) - 1, rg_end + half_rend_width)
                    set_org_range(end - max_rend_width, end, xmin, xmax)
                    last_range_val = range_val = round((end - max_rend_width) / half_rend_width)
                    return figure, range_val
            else:
                xminid, xmaxid = x_indexs[xmin], x_indexs[xmax]
            update_yaxis(figure, xminid, xmaxid)
        return figure, range_val

    app.run_server(mode=show_mode, height=height + 50)
