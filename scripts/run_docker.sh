#!/bin/bash

python -m banbot dbcmd --action=rebuild

python -m banbot spider & python -m banbot trade
