因最开始在美国VPS部署无法直接访问币安api，所以加了国内代理。这是支持打包代理的镜像版本。

```shell
# 在此目录编译镜像
docker build . -t python39

# 启动容器
docker run -d -v /root:/root --name highfreq --network=host --cap-add=NET_ADMIN python39

# 进入容器
docker exec -it highfreq /bin/bash
```
