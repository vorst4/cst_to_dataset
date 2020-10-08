from util.cst_to_dataset import cst_to_dataset
import os
import argparse

is_running_on_desktop = os.name == 'nt'

# on the server, the partition id is passed as an argument
if is_running_on_desktop:
    partition_id = 0
else:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition_id", help="server partition id", type=int)
    job_id = parser.parse_args().job_id
    partition_id = parser.parse_args().partition_id

cst_to_dataset(partition_id)
