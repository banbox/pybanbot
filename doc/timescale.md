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
```shell
vim /var/lib/pgsql/14/data/postgresql.conf
# 找到timezone和log_timezong，将值修改为UTC

# 进入数据库：
psql -U postgres -h localhost
select pg_reload_conf();
exit

# 初始化数据库表结构
python -m banbot dbcmd --force --action=rebuild -c /root/bantd/config.json

```
