# 基础镜像编译
```dockerfile
FROM python:3.11.6
RUN pip install --upgrade --force-reinstall -r requirements.txt
```
从上面内容生成不含源代码的基础镜像：  
`docker commit xxx anyongjin/banbot:3`

# 打包运行镜像
```shell
cd /root/banbot
docker build . -t banbot -f ./docker/Dockerfile
```

# 为用户启动机器人
创建用户目录:  
```shell
account='smith'
upath="/root/banusers/$account"
mkdir $upath && cd $upath
```

创建启动脚本(start.sh)：  
```shell
#!/bin/bash

account=$(basename "$PWD")
upath="/root/banusers/$account"
run_path="${upath}/run_docker.sh"
docker rm -f $account || true
docker run -d --name $account \
       --env-file ./env.txt \
       -e account=$account \
       -v /root:/root \
       --net=host \
       --entrypoint $run_path anyongjin/banbot:3
echo "$account start ok, run docker logs --tail 50 $account to view logs"
```

创建容器运行脚本(run_docker.sh)：  
```shell
#!/bin/bash
python --version
python -m banbot trade -c ~/ban_data/config.yml -c ~/ban_data/config.local.yml -c /root/banusers/$account/config.yml
```