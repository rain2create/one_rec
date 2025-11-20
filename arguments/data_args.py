# coding=utf8
# ========================================================
#   Copyright (C) 2025 All rights reserved.
#   
#   filename : data_args.py
#   author   : ***
#   date     : 2025-07-18
#   desc     : 
# ======================================================== 
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class DataArguments:
    # tokenizer
    codebook_layers: int = 3
    codebook_size: int = 1500
    remote_mapping_file : str = '/data01/search/lusitong/dev/onerec_tokenizor/data/clustering_results'
    mapping_file: str = '../data/url_mapping'
    bos_token_id : int = 0
    unk_token_id : int = 1501

    # dataset
    #root_dir: str ="/mnt/video.rank.s101"
    root_dir: str ="/mnt/video.rank.s101.pos_neg"
    corpus_days: int = 2
    with_today: bool = False
    batch_size: int = 1024
    # batch_size: int = 512
    shuffle_buffer_size: int = 10000
    compress_type: str = "gzip"
    done_spec: str = "onerec_sft_lazy_decoder_256_new_id"
    file_shuffle: bool = False
    num_workers: int = 20
    test_root_dirs: str = ""
    val_root_dirs: str = ""
    inter_file_shuffle = True
    reward_stage = False
    reward_batch_size = 20

    # parser
    # 默认参数陷阱，不要将可变对象作为默认参数
    columns_def =[
                ('f001.sign', "", 1),
                ('f002.scene', "", 0),
                ('f003.uid', "", 1),
                ('f004.url', "", 1),
                ('f005.duration', "0", 1),
                ('f006.rtype', "", 0),
                ('f007.author_id', "", 1),
                ('f008.videotag', "", 1),
                ('f009.vvctag', "", 0),
                ('f010.i_ent', "", 1),
                ('f011.reqtimes', "1", 1),
                ('f012.nbbo', "0", 0),
                ('f013.rank_score', "", 0),
                ('f014.model_name', "", 0),
                ('f015.logtime', "", 1),
                ('f016.auto_play', "1", 0),
                ('f017.sys_mute', "0", 0),
                ('f018.client_mute', "0", 0),
                ('f019.play_invisible', "0", 0),
                ('f020.user_effective', "0", 0),
                ('f021.is_finish_video', "0", 0),
                ('f022.is_auto_repeat', "0", 0),
                ('f023.is_manu_down', "0", 0),
                ('f024.manu_down_time', "0", 0),
                ('f025.is_manu_up', "0", 0),
                ('f026.manu_up_time', "0", 0),
                ('f027.is_close_tab', "0", 0),
                ('f028.close_tab_time', "0", 0),
                ('f029.like', "0", 0),
                ('f030.like_c', "0", 0),
                ('f031.dislike', "0", 0),
                ('f032.follow', "0", 0),
                ('f033.authorpanel', "0", 0),
                ('f034.videocollectpanel', "0", 0),
                ('f035.manu_right_click', "0", 0),
                ('f036.play_type', "", 0),
                ('f037.is_follow', "0", 1),
                ('f038.loglike', "", 0),
                ('f039.loglike_author_sim', "", 0),
                ('f040.loglike_itmask_sim', "", 0),
                ('f041.logsess', "", 0),
                ('f042.logsess_author_sim', "", 0),
                ('f043.logsess_itmask_sim', "", 0),
                ('f044.logsess_item_seqs', "", 1),
                ('f045.logsess_sim_seqs', "", 0),
                ('f046.logsess_time_seqs', "", 1),
                ('f047.logsess_att_seqs', "", 1),
                ('f048.logsess_ausim_seqs', "", 0),
                ('f049.likeauthLTsim_seqs', "", 1),
                ('f050.last_video', "", 0),
                ('f051.playletpanel', "", 0),
                ('f052.positive_item_seqs', "", 0),
                ('f053.negative_item_seqs', "", 0),
                ('f054.finish_next_list', "", 0),
                ('f055.gzhid_finish_next_url_list', "", 0),
                ('f056.gzh_ctr_grade', "", 0),
                ('f057.videotag_finish_next_url_list', "", 0),
                ('f058.videotag_ctr_grade', "", 0),
                ('f059.clStamp', "", 1),
                ('f060.collectionpanel', "", 0),
                ('f061.gdata', "", 0),
                ('f062.i_uf', "", 0),
                ('f063.optags', "", 1),
                ('f064.is_follow_store_optag', "", 1),
                ('f065.playlet_title', "", 1),
                ('f066.LongIdSeqs', "", 0),
                ('f067.LongScSeqs', "", 0),
                ('f068.cand_sturl', "", 0),
                ('f069.refer_scene', "3", 0),
                ("f070.multiScore", "", 0),
                ("f071.logsess_seqsDwell" ,"", 1),
                ("f072.logsess_seqsQid" , '', 1),
                ("f073.logsess_seqsDuration", "", 1),
                ("f074.logsess_seqsVideotag" ,"", 1),
                ("f075.logsess_seqsKws" ,"", 1),
                ("f076.vfi_md5_feature","", 0),
                ("f077.vfi_sign_scene_feature","", 0),
                ("f078.vfi_qid_feature","", 0),
                ("f079.logsess_seqsOptag","", 1),
                ("f080.logQue_seqsItem","", 1),
                ("f081.logQue_seqsTime","", 1),
                ("f082.logQue_seqsAtt","", 1),
                ("f083.logQue_seqsDwell","", 1),
                ("f084.logQue_seqsQid","", 1),
                ("f085.logQue_seqsDuration","", 1),
                ("f086.logQue_seqsVideotag","", 1),
                ("f087.logQue_seqsKws","", 1),
                ("f088.logQue_seqsOptag","", 1),
                ("f089.favorite","0", 0),
                ("f090.favorite_c","0", 0),
                ("f091.logreallike_seqsItem", "", 1),
                ("f092.logreallike_seqsTime", "", 0),
                ("f093.logreallike_seqsFunc", "", 0),
                ("f094.logreallike_seqsSignscene", "", 0),
                ("f095.logreallike_seqsQid", "", 1),
                ("f096.logreallike_seqsDuration", "", 1),
                ("f097.logreallike_seqsVideotag", "", 1),
                ("f098.logreallike_seqsKws", "", 1),
                ("f099.logreallike_seqsOptag", "", 1),
                ("f100.logdislike_seqsItem", "", 1),
                ("f101.logdislike_seqsTime", "", 0),
                ("f102.logdislike_seqsQid", "", 1),
                ("f103.logdislike_seqsDuration", "", 1),
                ("f104.logdislike_seqsVideotag", "", 1),
                ("f105.logdislike_seqsKws", "", 1),
                ("f106.logdislike_seqsOptag", "", 1),
                ("f107.qid_fans", "0", 1),
                ("f108.total_like", "0", 1),
                ("f109.logQue_seqsCount", "", 1),
                ("f110.logsess_seqsCount", "", 1),
                ("f111.likeauthLTList", "", 0),
                ("f112.likeauthLTWeight", "", 0),
                ("f113.realUTPTag", "", 0),
                ("f114.realUTPWeight", "", 0),
                ("f115.tj_url", "", 0),
                ("f116.reqid", "", 0),
                ("f117.mscWeight", "", 0),
                ("f118.nbbe", "", 0),
                ("f119.autok", "", 0),
                ("f120.optags_zh", "", 0),
                ("f121.logSess_interact_infos", "", 0),
                ("f122.logQue_interact_infos", "", 0),
                ("f123.logReallike_interact_infos", "", 0),
                ("f124.logDislike_interact_infos", "", 0),
                ("f125.is_likeauthLT_author", "", 0),
                ("f126.is_realUTP_tags", "", 0),
                ("f127.urlDycnts", "", 0),
                ("f128.last_positive_dwell_time", "", 0),
                ("f129.xsv_pop", "", 0),
                ("f130.store_s", "", 0),
                ("f131.vfmodel", "", 0),
    ]
    reward_serving_column_idx = [idx for idx, i in enumerate(columns_def) if i[2] == 1]

