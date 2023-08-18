#!/bin/bash

python -m banbot dbcmd --action=rebuild --yes

python -m banbot spider & python -m banbot trade
