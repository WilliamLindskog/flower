#!/bin/bash

declare -A data_list

# read all keys from treesXnets/data/constants.json
# and add only key to data_list, not the value

while IFS== read -r key value; do
    data_list[$key]=$value
done < <(jq -r 'to_entries[] | "\(.key)=\(.value)"' treesXnets/data/constants.json)

#print all keys
for model in mlp cnn; do 
    for key in "${!data_list[@]}"; do
        echo "$key"
        python -m treesXnets.main --config-path conf --config-name base model_name=${model} dataset.name=${key}
    done
done