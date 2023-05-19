#!/bin/bash

# 1. ���ҽ�����Ϊ"python -m banbot"�Ľ��̲�ɱ��
pid=$(ps aux | grep "python -m banbot" | head -n 1 | awk '{print $2}')
echo "killing bot: $pid"
kill $pid

# 2. �ݴ浱ǰĿ¼·��
tmp_path=$(pwd)

# 3. ���� /root/banbot Ŀ¼
cd /root/banbot

# 4. ִ�� git pull ��ȡ���´���
echo "git pulling..."
output=`git pull origin master`
echo $output
sleep 1

# 5. �ָ���ǰ·��
cd $tmp_path

# 6. �ٴ�����������
echo "starting bot..."
nohup python -m banbot trade --pairs BTC/TUSD -c /root/ban_data/config.json > /root/trade.out 2>&1 &
sleep 2
echo "$(tail -n 30 /root/trade.out)"

echo -e "\nTo show more log, use:"
echo "tail -30 /root/trade.out"
