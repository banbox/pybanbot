FROM python:3.11.6

COPY . /banbot
WORKDIR /banbot

RUN pip install --upgrade -r requirements.txt

RUN python setup.py install

# ENTRYPOINT ["/bin/bash", "scripts/run_docker.sh"]
