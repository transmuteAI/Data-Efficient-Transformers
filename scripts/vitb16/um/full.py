import os

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/full_um_vit.py' --name 'full_um_vit_pretrained' --num_steps {4725} --warmup_steps {225} --select_every {50} --eval_every {45} --pretrained --logger_dir './vit_logger/um/full' --gradient_accumulation_steps 5")