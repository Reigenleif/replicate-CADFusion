# set it to your data path
data_path=data/sl_data
# by default set it to CADFusion/exp
exp_path=exp/model_ckpt
# by default set it to CADFusion/data
vf_path=data/vf_data
train_data=$data_path/train.json
eval_data=$data_path/val.json

# This script requires your SL run named as xxxx0, because for each VF stage, the final digit increments 
# to show the number of VF rounds finished.
# e.g. SL name: CAD-0
#         base_name: CAD- (remove the last digit, the script autofills it)
#         VF run 1: CAD-1 (automatically)
#         VF run 2: CAD-2 (automatically)
#         ...
base_name=$1

run_name=${base_name}0
./scripts/generate_samples.sh $run_name test "--sample-len 100 --device-map auto"
./scripts/generate_samples.sh $run_name train "--sample-len 400 --device-map auto"