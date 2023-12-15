#!/bin/bash

# Dataset dict for number of classes
declare -A dataset_dict
dataset_dict=(["reg_cat_house_sales"]=1 ["reg_num_abalone"]=1 ["clf_num_credit"]=2 ["clf_cat_covertype"]=2)


# MLP Runs
for dataset in clf_cat_covertype
do 
    for partition in iid linear
    do
        for model_size in small medium large
        do
           num_classes=${dataset_dict[${dataset}]}
           python -m treesXnets.main --config-path conf --config-name mlp dataset.name=${dataset} dataset.num_classes=${num_classes} model.model_size=${model_size} dataset.partition=${partition}
        done
    done
done