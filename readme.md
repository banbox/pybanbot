# 策略和交易对
策略需继承自`BaseStrategy`，当某个策略应用到多个交易对时，每个交易对对应一个策略实例。  
一个交易对可以有多个不同的时间帧维度，一个策略只能对其中一个时间帧维度生效，不能同时作用于一个交易对的多个时间帧维度。  

# 上下文变量
交易对+时间帧 对应唯一一组上下文变量。即：一组上下文变量可能被多个策略同时使用，应该存储策略无关的信息。  
由于数据回调和订单状态更新回调都会随机不定时从异步协程中执行。
为避免访问上下文变量时混乱，需在外部使用`async with TempContext(...)`获取锁，方可访问上下文变量。

# 注意
关于数组的操作，尽量转成numpy，速度比pandas快很多倍。  
numpy中涉及循环的，尽量使用numba


# 性能测试优化
如需测试各个函数耗时，可在`cmds/entrys.py`中执行下面命令：
```python
import cProfile
cProfile.runctx('asyncio.run(backtesting.run())', globals(), locals(), sort='tottime')
```


# 部署
```shell
# 启动机器人
nohup python -m banbot trade -c /root/ban_data/config.json > /root/trade.out 2>&1 &

# 查看日志
tail -30 /root/trade.out

# 查看进程
ps aux | grep banbot
```

# TODO



