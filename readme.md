# 注意
关于数组的操作，尽量转成numpy，速度比pandas快很多倍。  
numpy中涉及循环的，尽量使用numba


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



