#!/bin/bash


# Default values
DEVICE=0
SEED=0
DOMAIN="walker"
TASK="walker_walk"
NUM_PRETRAIN_FRAMES=1000010
AGENT="dsquare"

# Other variables to formulate the directory name
TIME="$(date +'%H%M%S')"
DATE="$(date +'%Y.%m.%d')"

# Parse named parameters
while [ "$1" != "" ]; do
    case $1 in
        --device )          shift
                            DEVICE=$1
                            ;;
        --seed )            shift
                            SEED=$1
                            ;;
        --domain )          shift
                            DOMAIN=$1
                            ;;
        --task )            shift
                            TASK=$1
                            ;;
        --num_pretrain_frames ) shift
                            NUM_PRETRAIN_FRAMES=$1
                            ;;
        --agent )           shift
                            AGENT=$1
                            ;;
        * )                 echo "Invalid option: $1"
                            exit 1
    esac
    shift
done


# Run the pretraining
CUDA_VISIBLE_DEVICES=${DEVICE} python pretrain.py agent=${AGENT} domain=${DOMAIN} task=${TASK} num_train_frames=${NUM_PRETRAIN_FRAMES} seed=${SEED} hydra.run.dir=./online/${DATE}/${TIME}_${AGENT}_${TASK} &&

# Run the offline finetuning
CUDA_VISIBLE_DEVICES=${DEVICE} python offline.py snapshot_ts=0 agent=${AGENT} task=${TASK} seed=${SEED} replay_buffer_dir=./online/${DATE}/${TIME}_${AGENT}_${TASK} hydra.run.dir=./offline/${DATE}/${TIME}_${AGENT}_${TASK}



