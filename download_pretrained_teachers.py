# %%
import pymongo
from pymongo import MongoClient
import matplotlib.pyplot as plt
import pymongo
from bson.objectid import ObjectId
import pandas as pd
from munch import DefaultMunch
import os
import torch
import sys
import numpy as np

DB_URL = "mongodb://rk2:37373/?retryWrites=true&w=majority"
DB_NAME = "ast_dcase22t1"


mongodb_client = MongoClient(DB_URL)
mongodb = mongodb_client[DB_NAME]


def load_metric(_id, db=mongodb):
    if not isinstance(id, ObjectId):
        _id = ObjectId(_id)
    df = pd.DataFrame({k: v for k, v in db.metrics.find_one({"_id": _id}).items()
                       if k in {'steps',  'values'}})
    df = df.set_index("steps")
    return df


# %%
q = {"_id": {"$gte": 224, "$lte": 272},
     "config.basedataset.use_full_dev_dataset": 1}
cmds = []
reeval_cmds = []
for e in mongodb["runs"].find(q):
    print(e['_id'])
    out_file = f"teacher_models/passt_{e['_id']}.pt"
    if os.path.isfile(out_file):
        print("Already here: ", e['_id'])
        continue
    exp_name = e["experiment"]["name"]
    run_id = str(DB_NAME) + "_" + str(e['_id'])
    host_name = e['host']['hostname'].replace(
        "rechenknecht", "rk").replace(".cp.jku.at", "")
    ot = e["config"]["trainer"]['default_root_dir'].split(sep='/')[1]
    print(ot)
    output_dir = "dcase22/malach_dcase22/" + ot
    ckpts_path = f"/share/rk8/home/fschmid/deployment/{output_dir}/{exp_name}/{run_id}/checkpoints/"
    print(ckpts_path)
    assert os.path.isdir(ckpts_path)
    ckpt = ckpts_path + os.listdir(ckpts_path)[-1]
    print(f"Copying {ckpt}")
    ckpt = torch.load(ckpt)
    net_statedict = {k[4:]: v.cpu()
                    for k, v in ckpt['state_dict'].items() if k.startswith("net.")}

    torch.save(net_statedict, out_file)
    


    
# %%
