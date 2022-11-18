import os

os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {10} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {20} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {50} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")