# 贡献代码

# 项目结构介绍
[项目目录介绍](doc/introduce.md)  
[关于架构选型](doc/sys_structure.md)

# 开发环境配置
[安装时序数据库timescaledb](doc/timescale.md)  
redis: linux可直接安装，windows可通过docker安装。  
安装miniconda，选择python 3.10或3.11

# 数据抓取和维护
用到的数据包括：K线数据，订单簿，推送订单流，
公共实时数据由Spider负责抓取更新并通知bot执行

# 上下文变量
交易对+时间帧 对应唯一一组上下文变量。即：一组上下文变量可能被多个策略同时使用，应该存储策略无关的信息。  
由于数据回调和订单状态更新回调都会随机不定时从异步协程中执行。
为避免访问上下文变量时混乱，需在外部使用`async with TempContext(...)`获取锁，方可访问上下文变量。

# 注意
* 只在必要时使用异步，因伪异步增加了复杂性，在回测时性能较差
* 关于数组的操作，尽量转成numpy，速度比pandas快很多倍。  
* numpy中涉及循环的，尽量使用numba
* 对于list的筛选，尽量使用列表推导式，把所有条件放在一个if中完成，效率最高；效率次高的是逐个条件使用列表推导式


# 性能测试优化
如需测试各个函数耗时，可在`cmds/entrys.py`中执行下面命令：
```python
import cProfile
cProfile.runctx('asyncio.run(backtesting.run())', globals(), locals(), sort='tottime')
```


# 部署
```shell
# 启动机器人
nohup python -m banbot trade > /root/trade.out 2>&1 &

# 查看日志
tail -30 /root/trade.out

# 查看进程
ps aux | grep banbot
```