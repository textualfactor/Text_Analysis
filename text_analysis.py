#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:40:49 2020

@author: eklavya
"""
import sys
import threading 
sys.path.insert(0, './src/')

from tokenize_mdna import tokenize
from google_cluster import *
from process_cluster import cluster_process
from get_dic_bow import *
from SVD_doc_loading import * 
from SVD_topic_importance import *


wdir = './data/'
output = './output/output_all.csv'


cluster_size=50

t1 = threading.Thread(target=tokenize, args=(wdir, output)) 
t2 = threading.Thread(target=google_cluster, args=(cluster_size,)) 

t1.start() 
t2.start() 

t1.join()
t2.join()

cluster_process()

title = 'all'
csv_path = 'output/output_' + title + '.csv'
dic_path = title + '_dictionary'
npy_path = title + '_bag_words.npy'
dic_bow()
SVD_doc_load()
SVD_topic_importance()
