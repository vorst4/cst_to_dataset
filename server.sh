#!/usr/bin/bash

# verify that argument (partition id) is passed
if [ -z "$1" ]
then
  echo "ERROR: partition id needs to be passed as an argument"
  exit 1
fi

# verify that only 1 argument is passed
if [ $# -gt 1 ]
then
  echo "ERROR: $# arguments are given, only 1 is required"
fi

# verify that number of partitions is not > 3
if [ $1 -gt 3 ]
then
  echo "ERROR: partition id is > 3"
  exit 1
fi

# verify that number of partitions is >= 0
if [ $1 -lt 0 ]
then
  echo "ERROR: partition id is < 0"
  exit 1
fi

declare -a partitions=("tue.default.q"
                       "elec.default.q"
                       "elec.gpu.q"
                       "elec-em.gpu.q")

# execute jobs
export partition_id=$1
sbatch  --job-name=projects_to_dataset_$1 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=16 \
        --time=10-00:00:00 \
        --partition=${partitions[$1]} \
        --output=output \
        --error=error \
        --mail-user=d.m.n.v.d.vorst@student.tue.nl \
        --mail-type=ALL \
        task.sh