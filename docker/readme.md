���ʼ������VPS�����޷�ֱ�ӷ��ʱҰ�api�����Լ��˹��ڴ�������֧�ִ������ľ���汾��

```shell
# �ڴ�Ŀ¼���뾵��
docker build . -t python39

# ��������
docker run -d -v /root:/root --name highfreq --network=host --cap-add=NET_ADMIN python39

# ��������
docker exec -it highfreq /bin/bash
```
