# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : logger.py
#   author   : ***
#   date     : 2025-07-12
#   desc     : 
# ======================================================== 

import logging, time

cur_hour = time.strftime("%Y%m%d%H", time.localtime())
log_dir = "../log/"

logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename="%slog_%s" % (log_dir, cur_hour),
        filemode='a')
logger = logging.getLogger()