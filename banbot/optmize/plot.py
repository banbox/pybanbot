#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : plot.py
# Author: anyongjin
# Date  : 2023/2/21
import asyncio

import numpy as np
import pandas as pd
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


def _add_mark(fig, ts_col: pd.Series, time_ms: List[int], mtags: List[str], y_vals, marker=None, row=1):
    # time_ms是对应的bar的UTC时间
    min_ts, max_ts = ts_col.min(), ts_col.max()
    m_id_p = [(i, v) for i, v in enumerate(time_ms) if min_ts <= v <= max_ts]
    m_loc = []
    if m_id_p:
        m_loc, m_id = list(zip(*m_id_p))
        m_tag = list(map(lambda x: mtags[x], m_loc))
        mark_x, mark_y = list(m_id), np.array(y_vals)[list(m_loc)]
        ind = go.Scatter(x=mark_x, y=mark_y, mode='markers+text', text=m_tag, marker=marker)
        trade_args = dict()
        if row == 1:
            trade_args = dict(secondary_y=True)
        fig.add_trace(ind, row=row, col=1, **trade_args)
    return m_loc, m_id_p


def _add_order_ind(fig, df: pd.DataFrame, ind: dict):
    # 显示订单入场和退出；当前列是入场信号列，
    date_col = df['date']
    assert str(date_col.dtype.name).find('datetime') >= 0, '`date` col must be type: datetime'
    enter_at, enter_tag, enter_price = ind['enter_at'], ind['enter_tag'], ind['enter_price']
    exit_at, exit_tag, exit_price = ind['exit_at'], ind['exit_tag'], ind['exit_price']
    enter_price = np.array(enter_price)
    exit_price = np.array(exit_price)

    ent_color = ind.get('enter_color') or 'green'
    exit_color = ind.get('exit_color') or 'blue'
    in_loc, in_id_p = _add_mark(fig, date_col, enter_at, enter_tag, enter_price, dict(color=ent_color))
    out_loc, out_id_p = _add_mark(fig, date_col, exit_at, exit_tag, exit_price, dict(color=exit_color))

    # 计算可绘制的线段
    line_ids = set(in_loc).intersection(out_loc)
    lin_id_p = [(i, v) for i, v in in_id_p if i in line_ids]
    lout_id = [v for i, v in out_id_p if i in line_ids]
    if lin_id_p:
        # line_loc在in_id_p和out_id_p中应该完全相同
        line_loc, lin_id = list(zip(*lin_id_p))
        line_loc, lin_id = list(line_loc), list(lin_id)
        start_x, start_y = lin_id, enter_price[line_loc]
        end_x, end_y = lout_id, exit_price[line_loc]
        line_color = ind.get('line_color') or 'blue'
        line_args = dict(mode='lines', name='order', line=dict(color=line_color, width=2))
        for i in range(len(lin_id)):
            fig.add_trace(go.Scatter(x=[start_x[i], end_x[i]], y=[start_y[i], end_y[i]],
                                     **line_args), row=1, col=1, secondary_y=True)


def _add_indicator(fig, df, inds):
    x_labels = df['date']
    sub_yrange_calcs = []
    for ind in inds:
        name = ind.get('col')
        row_num = ind.get('row', 1)
        dtype = ind.get('type', 'scatter')
        func_args = dict(x=x_labels, name=name)
        if row_num == 1:
            func_args['hoverinfo'] = 'skip'
        if ind.get('color'):
            func_args['line'] = dict(color=ind.get('color'))
        if ind.get('opacity'):
            func_args['opacity'] = ind.get('opacity')
        if dtype == 'scatter':
            trace_func = go.Scatter
        elif dtype == 'mark':
            marker = dict(
                symbol=ind['symbol'],
                size=9,
                line=dict(width=1),
                color=ind.get('color'),
            )
            _add_mark(fig, df, ind['ids'], ind['tags'], ind['valy'], marker, row=row_num)
            continue
        elif dtype == 'order':
            _add_order_ind(fig, df, ind)
            continue
        elif dtype == 'bar':
            trace_func = go.Bar
        else:
            raise ValueError(f'unsupport trace type: {dtype}')
        trace_args = dict(row=row_num, col=1)
        if row_num == 1:
            trace_args['secondary_y'] = True
        else:
            sub_yrange_calcs.append((row_num, _make_yrange(df[name])))
        if not func_args.get('y'):
            func_args['y'] = df[name]
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
        # logger.warning('set org range: %s  %s', xmin, xmax)
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
            # logger.warning('xaxis:  %s  %s', xmin, xmax)
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


async def run_test():
    from banbot.storage.base import init_db, db
    from banbot.optmize.bt_analysis import BTAnalysis, order_plot_ind
    init_db(db_url='postgresql://postgres:123@[127.0.0.1]:5432/bantd')
    with db():
        backtest_dir = r'E:\trade\ban_data\backtest'
        btres = await BTAnalysis.load(backtest_dir)
        df = btres.load_df(start_ms=1681689600000, stop_ms=1681776000000)
        df['date'] = pd.to_datetime(df['date'], utc=True, unit='ms')
        od_ind = order_plot_ind(10)
    plot_fin(df, [od_ind], view_width=200)


if __name__ == '__main__':
    asyncio.run(run_test())
