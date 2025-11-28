#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : check_repeat.py
#   author   : ***
#   date     : 2025-11-26
#   desc     : 
# ======================================================== 
import sys


s = set()
a = 0
d = 0
for line in sys.stdin:
    secs = line.strip().split('\t')
    k = secs[1]
    if k in s:
        d += 1
    else:
        s.add(k)
    a += 1
    if a % 10000 == 0:
        print(d, a, d/a)

