#!/bin/bash

# 1. find process: "python -m banbot" and kill
pid_list=$(ps aux | grep "python -m banbot" | awk '{print $2}')

for pid in $pid_list; do
  echo "killing bot: $pid"
  kill $pid
  while kill -0 $pid 2> /dev/null; do
    sleep 0.3
  done
done

# 2. save current path temp
tmp_path=$(pwd)

# 3. get into /root/banbot
cd /root/banbot

# 4. exec: git pull
echo "git pulling..."
output=`git pull origin master`
echo $output
sleep 1

# 5. restore pwd from tmp
cd $tmp_path

# 6. restart the bot
echo "starting bot..."
nohup python -m banbot trade --pairs BTC/TUSD -c /root/ban_data/config.json > /root/trade.out 2>&1 &
sleep 2
echo "$(tail -n 30 /root/trade.out)"

echo -e "\nTo show more log, use:"
echo "tail -30 /root/trade.out"
