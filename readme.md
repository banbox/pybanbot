# ���Ժͽ��׶�
������̳���`BaseStrategy`����ĳ������Ӧ�õ�������׶�ʱ��ÿ�����׶Զ�Ӧһ������ʵ����  
һ�����׶Կ����ж����ͬ��ʱ��֡ά�ȣ�һ������ֻ�ܶ�����һ��ʱ��֡ά����Ч������ͬʱ������һ�����׶ԵĶ��ʱ��֡ά�ȡ�  

# �����ı���
���׶�+ʱ��֡ ��ӦΨһһ�������ı���������һ�������ı������ܱ��������ͬʱʹ�ã�Ӧ�ô洢�����޹ص���Ϣ��  
�������ݻص��Ͷ���״̬���»ص������������ʱ���첽Э����ִ�С�
Ϊ������������ı���ʱ���ң������ⲿʹ��`async with TempContext(...)`��ȡ�������ɷ��������ı�����

# ע��
��������Ĳ���������ת��numpy���ٶȱ�pandas��ܶ౶��  
numpy���漰ѭ���ģ�����ʹ��numba


# ���ܲ����Ż�
������Ը���������ʱ������`cmds/entrys.py`��ִ���������
```python
import cProfile
cProfile.runctx('asyncio.run(backtesting.run())', globals(), locals(), sort='tottime')
```


# ����
```shell
# ����������
nohup python -m banbot trade -c /root/ban_data/config.json > /root/trade.out 2>&1 &

# �鿴��־
tail -30 /root/trade.out

# �鿴����
ps aux | grep banbot
```

# TODO



