# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 8888 --nproc_per_node=1 \
# train.py --e ./examples/all
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --e ./examples/all_global
# CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --e ./examples/all
CUDA_VISIBLE_DEVICES=0 python train.py --e ./examples/all_quad_last