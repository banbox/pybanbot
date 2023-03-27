#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : plot.py
# Author: anyongjin
# Date  : 2023/2/21
import six
from datetime import datetime

from pandas import DataFrame
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Optional, Union


def _make_yrange(col):
    def get_xrange(startx, endx):
        coly = col.iloc[startx: endx]
        return coly.min(), coly.max()
    return get_xrange


def _add_indicator(fig, df, inds):
    import pandas as pd
    x_labels = df['date']
    sub_yrange_calcs = []
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
        elif dtype == 'order':
            # 显示订单入场和退出；当前列是入场信号列，
            enter_id, enter_tag = ind['enter_id'], ind['enter_tag']
            exit_id, exit_tag = ind['exit_id'], ind['exit_tag']
            # 过滤在当前df范围内的信号
            in_set = set(df.index.intersection(enter_id))
            out_set = set(df.index.intersection(exit_id))
            in_id_p = [(i, v) for i, v in enumerate(enter_id) if v in in_set]
            out_id_p = [(i, v) for i, v in enumerate(exit_id) if v in out_set]
            in_loc, in_id = list(zip(*in_id_p))
            out_loc, out_id = list(zip(*out_id_p))
            in_tag = [enter_tag[i] for i, v in enumerate(enter_id) if v in in_set]
            out_tag = [exit_tag[i] for i, v in enumerate(exit_id) if v in out_set]
            # 绘制入场信号和退出信号
            in_id, out_id = list(in_id), list(out_id)
            start_x, start_y = x_labels.loc[in_id], df.loc[in_id, 'close']
            end_x, end_y = x_labels.loc[out_id], df.loc[out_id, 'close']
            start = go.Scatter(x=start_x, y=start_y, mode='markers+text', text=in_tag, marker=dict(color='green'))
            end = go.Scatter(x=end_x, y=end_y, mode='markers+text', text=out_tag, marker=dict(color='blue'))
            fig.add_trace(start, row=1, col=1, secondary_y=True)
            fig.add_trace(end, row=1, col=1, secondary_y=True)
            # 计算可绘制的线段
            line_ids = set(in_loc).intersection(out_loc)
            lin_id = [v for i, v in in_id_p if i in line_ids]
            lout_id = [v for i, v in out_id_p if i in line_ids]
            start_x, start_y = x_labels.loc[lin_id], df.loc[lin_id, 'close']
            end_x, end_y = x_labels.loc[lout_id], df.loc[lout_id, 'close']
            line_args = dict(mode='lines', name='order', line=dict(color='blue', width=2))
            for i in range(len(lin_id)):
                fig.add_trace(go.Scatter(x=[start_x.iloc[i], end_x.iloc[i]], y=[start_y.iloc[i], end_y.iloc[i]],
                                         **line_args), row=1, col=1, secondary_y=True)
            return
        elif dtype == 'bar':
            trace_func = go.Bar
        else:
            raise ValueError(f'unsupport trace type: {dtype}')
        trace_args = dict(row=row_num, col=1)
        if row_num == 1:
            trace_args['secondary_y'] = True
        else:
            sub_yrange_calcs.append((row_num, _make_yrange(df[name])))
        fig.add_trace(trace_func(**func_args), **trace_args)
    return sorted(sub_yrange_calcs, key=lambda x: x[0])


def date_to_stamp(date, fmt: str = '%Y-%m-%d %H:%M:%S'):
    import calendar
    if isinstance(date, six.string_types):
        date = datetime.strptime(date, fmt)
    return calendar.timegm(date.timetuple())


def plot_fin(org_df: DataFrame, inds: Optional[List[Union[dict, str]]] = None, row_heights: Optional[List[float]] = None,
             height=None, width=None, view_width: int = 230, show_mode: str = 'inline'):
    '''
    显示蜡烛图和指标。默认绘制蜡烛图和交易量。
    :param org_df: 需要输出的数据
    :param inds: 指标列表[{col='volume', row=1, type='scatter|mark|bar', color='red', opacity=0.7, symbol=''}]
    :param row_heights: 多个子图的比例，需要和指标中最大row保持一致
    :param height:
    :param width:
    :param view_width: 显示的蜡烛数
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
    sub_yrange_calcs = []

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
        # 更新子图的纵坐标范围
        suby_ranges = dict()
        for row, calc in sub_yrange_calcs:
            ystart, yend = calc(xmin, xmax)
            if row in suby_ranges:
                ostart, oend = suby_ranges[row]
                suby_ranges[row] = min(ostart, ystart), max(oend, yend)
            else:
                suby_ranges[row] = ystart, yend
        for row in suby_ranges:
            fig.update_yaxes(range=suby_ranges[row], row=row)

    def make_graph(xmin, xmax):
        nonlocal sub_yrange_calcs
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
            sub_yrange_calcs = _add_indicator(fig, df, inds)
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
    from dash import html, dcc, Input, Output
    app = JupyterDash(__name__)
    figure = make_graph(0, view_width)
    # 设置range selector的提示
    range_style = dict(appearance='none', width='100%', background='#666', height='6px', borderRadius='3px')
    app.layout = html.Div(children=[
        dcc.Graph(id='graph1', figure=figure, config=dict(displayModeBar=False)),
        dcc.Input(id='range_sel', type='range', step=1, value=0, max=win_size - 1, style=range_style)
    ])
    x_indexs = [date_to_stamp(v) for v in x_labels.tolist()]
    item_delta = x_indexs[1] - x_indexs[0]
    last_range_val = 0

    def set_org_range(start, end, xmin: Optional[int] = None, xmax: Optional[int] = None):
        nonlocal x_labels, df, figure, x_indexs, rg_start, rg_end
        rg_start, rg_end = start, end
        df = org_df[start: end]
        # logger.warning(f'set org range: {xmin}  {xmax}')
        x_labels = df['date']
        x_indexs = [date_to_stamp(v) for v in x_labels.tolist()]
        if xmax is None:
            xmin, xmax = 0, min(view_width, end - start)
        else:
            xmin = next((i for i, v in enumerate(x_indexs) if v > xmin), 1) - 1
            xmax = next((i for i, v in enumerate(x_indexs) if v > xmax), 1)
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
            xmin = date_to_stamp(xmin[:xmin.find('.')])
            xmax = date_to_stamp(xmax[:xmax.find('.')])
            # logger.warning(f'xaxis:  {xmin}  {xmax}')
            if xmin < x_indexs[0] - item_delta:
                xminid, xmaxid = 0, min(len(x_indexs) - 1, view_width)
                if rg_start >= half_rend_width:
                    start = max(0, rg_start - half_rend_width)
                    set_org_range(start, start + max_rend_width, xmin, xmax)
                    last_range_val = range_val = round(start / half_rend_width)
                    return figure, range_val
            elif xmax > x_indexs[-1] + item_delta:
                xminid, xmaxid = max(0, len(x_indexs) - view_width), len(x_indexs) - 1
                if rg_end + half_rend_width <= len(org_df):
                    end = min(len(org_df) - 1, rg_end + half_rend_width)
                    set_org_range(end - max_rend_width, end, xmin, xmax)
                    last_range_val = range_val = round((end - max_rend_width) / half_rend_width)
                    return figure, range_val
            else:
                xminid = next((i for i, v in enumerate(x_indexs) if v > xmin), 1) - 1
                xmaxid = next((i for i, v in enumerate(x_indexs) if v > xmax), 1)
            update_yaxis(figure, xminid, xmaxid)
        return figure, range_val

    app.run_server(mode=show_mode, height=height + 50)
