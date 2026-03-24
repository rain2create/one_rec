#coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : t.py
#   author   : ***
#   date     : 2025-11-24
#   desc     : 
# ======================================================== 
import sys

cnts = [0] * 20
for line in sys.stdin:
    secs = line.strip().split(',')
    cnts[0] += int(secs[29] == '1')
    cnts[1] += int(secs[30] == '1')
    cnts[2] += int(secs[31] == '1')
    cnts[3] += int(secs[32] == '1')
    cnts[4] += int(secs[33] == '1')
    cnts[5] += int(secs[50] == '1')
    cnts[6] += int(secs[89] == '1')
    cnts[7] += int(secs[90] == '1')
    cnts[8] += int(secs[22] == '1')
    cnts[9] += int(secs[23] == '1')
    cnts[10] += int(secs[49] == '1')
    cnts[11] += int(secs[20] == '1')
    cnts[12] += int(secs[28] == '1')
    cnts[13] += int(secs[88] == '1')
    cnts[14] += int(secs[49] == '1' and secs[20] == '1')
    cnts[15] += int(float(secs[23]) < 2.0 and secs[22] == '1')
    cnts[16] += 1
print(list(zip(list(range(len(cnts))), cnts)))

