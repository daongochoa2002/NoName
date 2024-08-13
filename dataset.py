# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import math
import copy
import time
import numpy as np
from random import shuffle
from collections import defaultdict

class Dataset:
    """Implements the specified dataloader"""
    def __init__(self,
                 ds_name):
        """
        Params:
                ds_name : name of the dataset
        """
        self.name = ds_name
        # self.ds_path = "<path-to-dataset>" + ds_name.lower() + "/"
        self.ds_path = "datasets/" + ds_name.lower() + "/"
        self.ent2id = {}
        self.rel2id = {}
        self.data = {"train": self.readFile(self.ds_path + "train.txt"),
                     "valid": self.readFile(self.ds_path + "valid.txt"),
                     "test":  self.readFile(self.ds_path + "test.txt")}


        self.start_batch = 0
        self.all_facts_as_tuples = None

        self.convertTimes()

        self.all_facts_as_tuples = set([tuple(d) for d in self.data["train"] + self.data["valid"] + self.data["test"]])

        for spl in ["train", "valid", "test"]:
            self.data[spl] = np.array(self.data[spl]).astype(int)
        self.skip_dict = self.get_skipdict(self.data['train'].tolist()+self.data['valid'].tolist() + self.data['test'].tolist())
    def readFile(self,
                 filename):

        with open(filename, "r") as f:
            data = f.readlines()

        facts = []
        for line in data:
            elements = line.strip().split("\t")

            head_id =  self.getEntID(elements[0])
            rel_id  =  self.getRelID(elements[1])
            tail_id =  self.getEntID(elements[2])
            timestamp = elements[3]

            facts.append([head_id, rel_id, tail_id, timestamp])

        return facts


    def convertTimes(self):
        """
        This function spits the timestamp in the day,date and time.
        """
        for split in ["train", "valid", "test"]:
            for i, fact in enumerate(self.data[split]):
                fact_date = fact[-1]
                self.data[split][i] = self.data[split][i][:-1]
                date = list(map(float, fact_date.split("-")))
                self.data[split][i] += date



    def numEnt(self):

        return len(self.ent2id)

    def numRel(self):

        return len(self.rel2id)


    def getEntID(self,
                 ent_name):

        if ent_name in self.ent2id:
            return self.ent2id[ent_name]
        self.ent2id[ent_name] = len(self.ent2id)
        return self.ent2id[ent_name]

    def getRelID(self, rel_name):
        if rel_name in self.rel2id:
            return self.rel2id[rel_name]
        self.rel2id[rel_name] = len(self.rel2id)
        return self.rel2id[rel_name]

    @staticmethod
    def get_reverse_quadruples_array(quadruples, num_r):
        quads = np.copy(quadruples)
        quads_r = np.zeros_like(quads)
        quads_r[:, 1] = num_r + quads[:, 1]
        quads_r[:, 0] = quads[:, 2]
        quads_r[:, 2] = quads[:, 0]
        quads_r[:, 3] = quads[:, 3]
        quads_r[:, 4] = quads[:, 4]
        quads_r[:, 5] = quads[:, 5]
        return np.concatenate((quads, quads_r),axis=1).reshape(int(quads_r.shape[0] * 2),6)
    def get_skipdict(self, quadruples):
        """Used for time-dependent filtered metrics.
        return: a dict [key -> (entity, relation, timestamp),  value -> a set of ground truth entities]
        """
        filters = defaultdict(set)
        for src, rel, dst, year, month, day in quadruples:
            filters[(src, rel)].add(dst)
            filters[(dst, rel+self.numRel())].add(src)
        return filters

class QuadruplesDataset(Dataset):
    def __init__(self, quadruples, dataset, dataset_type='train'):
        self.quadruples = quadruples
        self.PAD_TIME = -1
        self.num_r = dataset.numRel()
        self.dataset_type = dataset_type
        self.dataset = dataset
    def __len__(self):
        return len(self.quadruples)

    def __getitem__(self, idx):
        quad = self.quadruples[idx]
        head_entity, relation, tail_entity, year, month, day = quad[0], quad[1], quad[2], quad[3], quad[4], quad[5]
        neg = np.random.randint(self.dataset.numEnt(), size=(500))
        return head_entity, relation, tail_entity, year, month, day, neg