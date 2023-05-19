#!/bin/bash

# 1. 查找进程名为"python -m banbot"的进程并杀死
pid=$(ps aux | grep "python -m banbot" | head -n 1 | awk '{print $2}')
echo "killing bot: $pid"
kill $pid

# 2. 暂存当前目录路径
tmp_path=$(pwd)

# 3. 进入 /root/banbot 目录
cd /root/banbot

# 4. 执行 git pull 拉取最新代码
echo "git pulling..."
output=`git pull origin master`
echo $output
sleep 1

# 5. 恢复当前路径
cd $tmp_path

# 6. 再次启动机器人
echo "starting bot..."
nohup python -m banbot trade --pairs BTC/TUSD -c /root/ban_data/config.json > /root/trade.out 2>&1 &
sleep 2
echo "$(tail -n 30 /root/trade.out)"

echo -e "\nTo show more log, use:"
echo "tail -30 /root/trade.out"
