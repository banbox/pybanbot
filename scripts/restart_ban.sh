#!/bin/bash

# 1. find process: "banbot" and kill
pkill -15 -f banbot

tmp_path=$(pwd)
cd /root/banbot

# 4. exec: git pull
echo "git pulling..."
output=`git pull origin master`
echo $output
sleep 1
cd $tmp_path

# 6. restart the bot
echo "starting bot..."
nohup python -m banbot trade --pairs BTC/TUSD -c /root/ban_data/config.json > /root/trade.out 2>&1 &
sleep 2
echo "$(tail -n 30 /root/trade.out)"

echo -e "\nTo show more log, use:"
echo "tail -30 /root/trade.out"
