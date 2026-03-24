# coding=utf8
# ========================================================
#   Copyright (C) 2019 All rights reserved.
#
#   filename : opstate_pika.py
#   author   : ***
#   date     : 2019-04-28
#   desc     :
# ========================================================
import sys
import socket
import hashlib
import redis

# pika
pdate_dup_pika_conf = {
        "zzzc2": {"ip": "10.175.162.34", "port": "8648", "auth": "3fe885fb4cec4dcd"},
        "bjmd": {"ip": "10.220.192.39", "port": "8648", "auth": "3fe885fb4cec4dcd"},
        "bjpdc": {"ip": "10.224.144.15", "port": "8648", "auth": "3fe885fb4cec4dcd"}
}

class OpstatePika:
    def __init__(self):
        jifang = "zzzc2"
        ip   = pdate_dup_pika_conf[jifang]["ip"]
        port = pdate_dup_pika_conf[jifang]["port"]
        auth = pdate_dup_pika_conf[jifang]["auth"]
        self.redis = redis.Redis(host = ip, port = int(port), db = 0, password = auth)

    def hmget_by_md5list(self, md5list, fields):
        with self.redis.pipeline(transaction=False) as p:
            for md5 in md5list:
                p.hmget(md5, fields)
            ret = p.execute()
        return ret

if __name__ == "__main__":
    op_pika = OpstatePika()
    md5s, allinfo = [], []
    cashe = []
    f_out = open("../data/clustering_results_dupid", "w")
    for line in open("../data/clustering_results", "r"):
        md5s.append(line.strip().split("\t")[0])
        allinfo.append(line.strip())
        if len(md5s) >= 10000:
            ret = op_pika.hmget_by_md5list(md5s, ["dupid"])
            gnids = [x[0].decode() if x and len(x) == 1 and x[0] else "" for x in ret]
            f_out.write("\n".join([x[0]+"\t"+x[1]for x in zip(allinfo, gnids)]) + "\n")
            #print("\n".join([x[0]+"\t"+x[1]for x in zip(allinfo, gnids)]))
            md5s, allinfo = [], []
    ret = op_pika.hmget_by_md5list(md5s, ["dupid"])
    gnids = [x[0].decode() if x and len(x) == 1 and x[0] else "" for x in ret ]
    f_out.write("\n".join([x[0]+"\t"+x[1]for x in zip(allinfo, gnids)]) + "\n")
    #print("\n".join([x[0]+"\t"+x[1]for x in zip(allinfo, gnids)]))
