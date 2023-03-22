#!/usr/bin/python3
# -*- coding: utf-8 -*-
# File  : inds.py
# Author: anyongjin
# Date  : 2023/3/2
from banbot.bar_driven.ind_common import *


class SMA(BaseInd):

    def __init__(self, period: int):
        super(SMA, self).__init__(period)
        self.div_key = f'c_div{period}'
        self._add_dep(self.div_key)

    def _compute(self, arr: np.ndarray) -> np.float64:
        return sum(arr[:, use_id_map.get()[self.div_key]])


class TR(BaseInd):
    '''
    经典的True Range指标。当前蜡烛和上一个蜡烛的真实波动幅度。反应2窗口的变动幅度。
    可用于动态价格止损。
    缺点：对于芝麻开花节节高明显趋势，会认为没有波动
    '''
    def __init__(self, *args):
        super(TR, self).__init__(2)

    def _compute(self, arr: np.ndarray) -> np.float64:
        crow = arr[-1, :]
        if arr.shape[0] < 2:
            return crow[1] - crow[2]
        prow = arr[-2, :]
        return max(crow[1] - crow[2], abs(crow[1] - prow[3]), abs(crow[2] - prow[3]))


class NATR(BaseInd):
    '''
    归一化的平滑价格波动。可衡量2个蜡烛内价格波动率。
    '''
    def __init__(self, period: int = 3):
        super(NATR, self).__init__(period)
        self.dep_key = 'tr'
        self._add_dep(self.dep_key, long_price_range)

    def _compute(self, arr: np.ndarray) -> np.float64:
        prow = arr[-2, :]
        tr_idx = use_id_map.get()[self.dep_key]
        if np.isnan(prow[self.out_idx]):
            atr_val = sum(arr[:, tr_idx]) / self.roll_len
        else:
            atr_val = (prow[self.out_idx] * (self.roll_len - 1) + arr[-1, tr_idx]) / self.roll_len
        return atr_val / long_price_range.val


class TRRoll(BaseInd):
    '''
    n窗口的TrueRange。能反应n窗口的价格变化幅度(默认n=4)
    无需再使用ATR计算TRRoll，因为窗口本身已经有平滑的作用。
    不可用于不同交易对之间比较。
    '''
    def __init__(self, period: int = 4):
        super(TRRoll, self).__init__(period)

    def _compute(self, arr: np.ndarray) -> np.float64:
        high_col, low_col = arr[:, 1], arr[:, 2]
        max_id = np.argmax(high_col)
        roll_max = high_col[max_id]
        min_id = np.argmin(low_col)
        roll_min = low_col[min_id]

        prev_tr = roll_max - roll_min
        if self.roll_len - min(max_id, min_id) <= 2:
            # 如果是从最后两个蜡烛计算得到的，则是重要的波动范围。
            return prev_tr
        # 从前面计算的TrueRange加权缩小，和最后两个蜡烛的TrueRange取最大值
        prev_tr *= (min(max_id, min_id) + 1) / (self.roll_len * 2) + 0.5
        return max(prev_tr, np.max(high_col[-2:]) - np.min(low_col[-2:]))


class NTRRoll(BaseInd):
    '''
    归一化，带窗口的TrueRange。可用于不同交易对之间比较。
    因分母（价格区间）可能剧烈变化，使用时最好
    '''
    def __init__(self, roll_size: int = 4):
        '''
        :param roll_size: 平滑窗口大小
        '''
        super(NTRRoll, self).__init__()
        self.dep_key = f'tr_roll{roll_size}'
        self._add_dep(self.dep_key, long_price_range)

    def _compute(self, arr: np.ndarray) -> np.float64:
        dep_id = use_id_map.get()[self.dep_key]
        return arr[-1, dep_id] / long_price_range.val


class NVol(BaseInd):
    '''
    倍数归一的成交量。可横向比较
    '''
    def __init__(self):
        super(NVol, self).__init__()
        self._add_dep(long_vol_avg)

    def _compute(self, arr: np.ndarray) -> np.float64:
        return arr[-1, 4] / long_vol_avg.val


''' 注册所有指标类到_ind_cls_map
**********************************************
'''
_clsmembers = inspect.getmembers(sys.modules[__name__], inspect.isclass)
for (cname, cls) in _clsmembers:
    if issubclass(cls, BaseInd) and cname != 'BaseInd':
        ind_call_map[to_snake_case(cls.__name__)] = cls


def _make_sub_ma120_calc():
    return make_ind_compute(SMA(120), lambda val, arr: arr[-1, 3] - val)


long_sub_ma120 = LongVar('long_sub_ma120', 900, 600, make_calc=_make_sub_ma120_calc)


def _make_atr_low_avg():
    natr = NATR()

    def handle_val(val, arr):
        ntr_p200 = arr[-600:, natr.out_idx]
        return avg_in_range(ntr_p200, 0.1, 0.4)
    return make_ind_compute(natr, handle_val)


long_atr_low = LongVar('long_atr_low', 600, 600, make_calc=_make_atr_low_avg)


''' 注册依赖于指标的自定义中间数据计算器
**********************************************
'''


# def make_atr_roll(args: str):
#     period = int(args)
#     return _make_ind_compute(ATR(period, roll=True), lambda x: x)
#
#
# ind_call_map.update(atr_roll=ATR)


def reset_ind_context(col_num: int):
    # 重置上下文信息
    use_cols.set([])
    use_refs.set(dict())
    use_id_map.set(dict())
    bar_num.set(0)
    inrows_num.set(100)
    dep_handlers.clear()
    BaseInd._instances.clear()
    BaseInd.col_offset = col_num
    logger.warning('context reseted!')


def set_use_inds(ind_objs: List[BaseInd]):
    '''
    使用一套新的指标组合计算。设置关心的指标列表
    【注意】请在实例化指标前，调用reset_ind_context方法
    :param ind_objs: 生成指标的函数
    :return:
    '''
    use_ind_objs.set(ind_objs)


def apply_inds_to_first(row: np.ndarray) -> np.ndarray:
    '''
    将一系列指标应用到OHLC的第一行数据上，输入一维数组，返回的二维数组，列长度有变化。
    :param row:
    :return:
    '''
    assert len(row.shape) == 1 and row.shape[0] >= 5, 'arr shape must be [n] (n>=5)'
    bar_num.set(1)
    # 开始计算指标
    arr = np.expand_dims(row, axis=0)
    long_price_range.on_bar(arr)
    long_bar_avg.on_bar(arr)
    long_vol_avg.on_bar(arr)
    long_sub_ma120.ensure_func()
    long_sub_ma120.on_bar(arr)
    long_atr_low.ensure_func()
    long_atr_low.on_bar(arr)
    for item in use_ind_objs.get():
        arr = item.calc_first(arr)
    return arr


def apply_inds(arr: np.ndarray) -> np.ndarray:
    '''
    将一系列指标应用到数组的最后一行数据上。
    :param arr: 必须是二维数组[[open,high,low,close,volume,count,long_vol]]
    :return:
    '''
    bar_num.set(bar_num.get() + 1)
    assert arr is not None and len(arr.shape) == 2 and arr.shape[0] > 1, \
        f'array shape must be [>2, n], current: {arr.shape}'
    assert BaseInd.col_offset > 5, 'BaseInd.col_offset is not initialized!'
    if not len(arr):
        return arr
    long_price_range.on_bar(arr)
    long_bar_avg.on_bar(arr)
    long_vol_avg.on_bar(arr)
    long_sub_ma120.on_bar(arr)
    long_atr_low.on_bar(arr)
    ohcl_arr.set(arr)
    [item() for item in use_ind_objs.get()]
    out_arr = ohcl_arr.get()
    ohcl_arr.set(None)
    return out_arr


def calc_ind_cols(arr: np.ndarray, use_inds: List[BaseInd]) -> np.ndarray:
    '''
    对数组所有行计算给定指标的值。（供测试用）
    :param arr:
    :param use_inds:
    :return:
    '''
    set_use_inds(use_inds)
    result = apply_inds_to_first(arr[0])
    pad_len = result.shape[1] - arr.shape[1]
    arr = append_nan_cols(arr, pad_len)
    for i in range(1, len(arr)):
        result = np.append(result, np.expand_dims(arr[i, :], axis=0), axis=0)
        result = apply_inds(result)
    return result


def calc_ind_col(arr: np.ndarray, ind_name: str) -> np.ndarray:
    '''
    计算单个指标的列值（供测试用）
    :param arr:
    :param ind_name:
    :return:
    '''
    call_func, cls_key = get_ind_callable(ind_name)
    assert call_func, f'unknown ind: {ind_name}'
    arg_text = ind_name[len(cls_key):]
    if inspect.isclass(call_func) and issubclass(call_func, BaseInd):
        reset_ind_context(arr.shape[1])
        use_inds = [call_func.get_by_str(cls_key, arg_text)]
    else:
        raise ValueError(f'unknown ind: {ind_name}')
    result = calc_ind_cols(arr, use_inds)
    return result[:, use_inds[0].out_idx]
