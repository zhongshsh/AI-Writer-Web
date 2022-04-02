#!/bin/sh
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt
pip uninstall -y oneflow
pip install --pre oneflow -f https://staging.oneflow.info/branch/master/cu112

