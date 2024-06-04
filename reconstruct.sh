CUDA_VISIBLE_DEVICES=0 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 0 -num 8 --skip &
CUDA_VISIBLE_DEVICES=1 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 1 -num 8 --skip &
CUDA_VISIBLE_DEVICES=2 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 2 -num 8 --skip &
CUDA_VISIBLE_DEVICES=3 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 3 -num 8 --skip &
CUDA_VISIBLE_DEVICES=4 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 4 -num 8 --skip &
CUDA_VISIBLE_DEVICES=5 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 5 -num 8 --skip &
CUDA_VISIBLE_DEVICES=6 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 6 -num 8 --skip &
CUDA_VISIBLE_DEVICES=7 python reconstruct.py -e examples/all_computed_rot_quad_last -c 18 --split examples/splits/shapenet_selected.json -d data -idx 7 -num 8 --skip &

