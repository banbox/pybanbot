FROM anyongjin/banbot:3
#RUN apt-get update
#RUN apt-get upgrade -qy
#RUN apt-get install iptables redsocks curl wget lynx -qy
#RUN update-alternatives --set iptables /usr/sbin/iptables-legacy
#RUN update-alternatives --set ip6tables /usr/sbin/ip6tables-legacy
#COPY redsocks.conf /etc/redsocks.conf
#COPY run.sh /tmp/proxy_all.sh


COPY . /banbot
WORKDIR /banbot
RUN pip install . --no-deps

# ENTRYPOINT ["/bin/bash", "scripts/run_docker.sh"]
