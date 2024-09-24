# Set common variables
model="..."

python main.py \
--model $model \
--prune_method "cfsp" \
--pruning_ratio 0.2  \
--nsamples 128 \
--a 1  \
--b 1  \
--c 1  \
--global_metrics angular \
--local_metrics three_w_one_wa \
--save_model "/var/s3fs-hgd/.../" \
--eval \
--cuda_friendly \
