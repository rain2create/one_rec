ps aux | grep seq_enc.finish.expl | awk '{print $2}' | xargs kill
