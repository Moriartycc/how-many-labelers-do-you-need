import torch
import numpy as np
import pandas as pd
from crowdkit.aggregation import DawidSkene
from crowdkit.aggregation import MajorityVote
from crowdkit.aggregation import GLAD
from torch.serialization import add_safe_globals

def aggregate():
    raw_votes = torch.load('raw_votes.pt', weights_only=False)['raw_votes']
    task_votes = torch.load('raw_votes.pt', weights_only=False)['task_votes']
    # print(task_votes)

    # Majority vote method
    MV_labels = MajorityVote().fit(task_votes)

    # Dawid Skene method
    DS_labels = DawidSkene(n_iter=20).fit(task_votes)

    # Generative model of Labels, Abilities, and Difficulties
    GLAD_labels = GLAD(n_iter=20, m_step_max_iter=3, silent=False).fit(task_votes)

    MV_labels.labels_.to_csv('MV.csv')
    MV_labels.probas_.to_csv('MV_probas.csv')
    DS_labels.labels_.to_csv('DS.csv')
    DS_labels.probas_.to_csv('DS_probas.csv')
    GLAD_labels.labels_.to_csv('GLAD.csv')
    GLAD_labels.probas_.to_csv('GLAD_probas.csv')
