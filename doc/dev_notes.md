### asyncio异步调度介入
在之前版本中，实现了一个模块模拟asyncio的异步时间流逝，以便在回测时使用await asyncio.sleep来模拟实时交易。  
但因为异步调度开销较大，性能较差，已不再使用此方案进行回测。  
如需查看相关代码，可找到提交:`bcb849211a85a467a19fc1ba0938eba6d3736a11`，然后查看`core/async_events.py`

## 日志文件过大怎么办？
linux中可使用logrotate限制日志文件大小。
[链接](https://linux.die.net/man/8/logrotate)
修改配置文件：
```shell
vim /etc/logrotate.conf
```
将`/root/logs`下的所有日志文件，最大限制100m，最多4个文件。
```text
/root/logs/* {
    size 100M
    create 644 root root
    rotate 4
}
```
使配置文件生效：
```shell
logrotate /etc/logrotate.conf
```
