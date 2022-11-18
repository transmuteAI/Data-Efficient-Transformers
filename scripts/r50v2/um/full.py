import os

os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/ultramnist/full_um_r50v2.py' --name 'full_um_r50v2_pretrained' --num_steps {4725} --warmup_steps {225} --select_every {50} --eval_every {45} --pretrained --logger_dir './r50v2_logger/um/full' --gradient_accumulation_steps 2")