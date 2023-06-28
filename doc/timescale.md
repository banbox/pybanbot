# 安装
必须使用CentOS 7，测试CentOS8安装时遇到很多不兼容错误。
可直接按照官方文档中Red Hat 7的安装指引：
Timescale Documentation | Install TimescaleDB on Linux
# Docker
```shell
docker pull timescale/timescaledb:latest-pg15

# 启动容器：
docker run -d --name timescaledb -p 5432:5432 -v /opt/pgdata:/home/postgres/pgdata/data -e POSTGRES_PASSWORD=Uf6CVdsZ3Dc timescale/timescaledb:latest-pg15

# 进入数据库：
docker exec -it timescaledb psql -U postgres -h localhost [-d bantd]

# 创建数据库
CREATE database bantd;
```

# 部署初始化
## 修改时区为UTC
```shell
# 进入数据库容器
docker exec -it timescaledb /bin/bash
# 执行下面命令，检查时区是否为UTC
# 如果不是docker安装，路径可能为： /var/lib/pgsql/14/data/postgresql.conf
cat /var/lib/postgresql/data/postgresql.conf|grep timezone
# 这里应该默认为UTC，如果不是UTC，则按下面修改：
exit  # 退出容器
docker cp timescaledb:/var/lib/postgresql/data/postgresql.conf ~/download/postgresql.conf
vim ~/download/postgresql.conf
# 找到timezone和log_timezong，将值修改为UTC
# 将文件复制到容器
docker cp ~/download/postgresql.conf timescaledb:/var/lib/postgresql/data/postgresql.conf

# 重新加载配置：
docker exec -it timescaledb psql -U postgres -h localhost
select pg_reload_conf();
exit
```
## 初始化数据库表结构
```shell
python -m banbot dbcmd --force --action=rebuild -c /root/bantd/config.json
```