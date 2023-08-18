# 使用Docker Compose运行机器人

## VPS要求
硬件：最低1核1G  
系统：CentOS 8最佳，7也可，其他linux系统也可。（Windows也可以，不过需要修改`compose.yml`，请自行修改）  
网络：选择离交易所最近的地域最佳，大部分加密货币交易所在日本东京。  

## 1. 拉取项目文件
需要将`banbot`和`ban_data`放在`/root`文件夹下：  
```shell
cd /root
git clone git@gitee.com:anyongjin/banbot.git
git clone git@gitee.com:anyongjin/ban_data.git
```
如果已下载过，执行下面命令更新即可：
```shell
cd /root/banbot && git pull origin master
cd /root/ban_data && git pull origin master
```
> 注意，如果修改了代码文件，执行`git pull`的时候可能会报冲突错误，如果不需要保留本地更改，可执行`git checkout -f HEAD`来舍弃更改，然后重新执行`git pull origin master`即可。  
> 如果需要保留更改，请自行解决冲突，然后commit文件
## 2. 修改配置&上传策略
```shell
cd /root/ban_data
cp config.json config.local.json
# config.local.json和config.json的结构相同，会覆盖config.json的同名配置。
# config.local.json不会被git追踪，适合存放敏感信息，如：交易所秘钥、微信通知token、策略名、时间周期、杠杆等
vim config.local.json
mkdir /root/ban-strategies
# 通过WinScp将策略文件上传到/root/ban-strategies
```
## 3. 安装Docker
[CentOS安装Docker](https://docs.docker.com/engine/install/centos/#install-using-the-repository)
> 如果遇到错误“Failed to synchronize cache for repo 'AppStream', ignoring this repo.”，需要把`/etc/yum.repo.d`下，对应的baseurl取消注释，域名改为vault.centos.org [参考](https://forums.centos.org/viewtopic.php?t=78708)
## 4. 启动机器人
```shell
cd /root/banbot
# 停止并删除旧的容器
docker compose stop && docker compose rm -f
# 重新编译镜像
docker compose build
# 启动所有服务
docker compose up -d

# 启动后可通过下面命令查看机器人日志：
docker compose logs --tail 30 bot
```
> 如果执行`docker compose build`时出现错误：symbol lookup error: runc: undefined symbol: seccomp_api_get，则需要执行`yum install libseccomp-devel`
