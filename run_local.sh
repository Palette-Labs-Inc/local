#!/bin/bash

source activate nosh
streamlit run ecviz/main.py --server.runOnSave True --server.allowRunOnSave True --server.headless True