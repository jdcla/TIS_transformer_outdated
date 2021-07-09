#!/bin/bash
THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float32 python TISRover_eval.py