#!/bin/bash
PROXY_SERVER="127.0.0.1"
PROXY_PORT=7890
echo "Configuration:"
echo "PROXY_SERVER=$PROXY_SERVER"
echo "PROXY_PORT=$PROXY_PORT"
echo "Setting config variables"
sed -i "s/vPROXY-SERVER/$PROXY_SERVER/g" /etc/redsocks.conf
sed -i "s/vPROXY-PORT/$PROXY_PORT/g" /etc/redsocks.conf
echo "Restarting redsocks and redirecting traffic via iptables"
/etc/init.d/redsocks restart
iptables -t nat -A OUTPUT -p tcp --dport 80 -j REDIRECT --to-port 12345
iptables -t nat -A OUTPUT -p tcp --dport 443 -j REDIRECT --to-port 12345
iptables -t nat -A OUTPUT -p tcp --dport 9443 -j REDIRECT --to-port 12345

# сю╬ц╣х╢Щ
tail -f /dev/null
