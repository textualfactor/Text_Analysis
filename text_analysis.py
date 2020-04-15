#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:40:49 2020

@author: eklavya
"""
import sys
sys.path.insert(0, './src/')

wdir = './data/'
output = 'output/output_all.csv'

import sys
from tokenize_mdna import tokenize
from google_cluster import google_cluster
from process_cluster import cluster_process



cluster_size=50

tokenize(wdir, output)
google_cluster(cluster_size)
cluster_process()