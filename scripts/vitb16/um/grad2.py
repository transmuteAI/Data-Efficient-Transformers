import os

# os.system(f"CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=12-23 python train13.py --cfg './configs/ultramnist/grad_um_vit.py' --name 'grad_um_vit_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {10} --eval_every {50} --pretrained --logger_dir './vit_logger/um/grad' --gradient_accumulation_steps 5")

# os.system(f"CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=12-23 python train13.py --cfg './configs/ultramnist/grad_um_vit.py' --name 'grad_um_vit_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {20} --eval_every {50} --pretrained --logger_dir './vit_logger/um/grad' --gradient_accumulation_steps 5")

# os.system(f"CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=12-23 python train13.py --cfg './configs/ultramnist/grad_um_vit.py' --name 'grad_um_vit_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {50} --eval_every {50} --pretrained --logger_dir './vit_logger/um/grad' --gradient_accumulation_steps 5")

# os.system(f"CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=12-23 python train13.py --cfg './configs/ultramnist/grad_um_vit.py' --name 'grad_um_vit_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {10} --eval_every {50} --pretrained --logger_dir './vit_logger/um/grad' --gradient_accumulation_steps 5")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_vit.py' --name 'grad_um_vit_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {20} --eval_every {50} --pretrained --logger_dir './vit_logger/um/grad' --gradient_accumulation_steps 5")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/rand/rand_um_vit.py' --name 'rand_um_vit_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {20} --eval_every {36} --pretrained --logger_dir './vit_logger/um/rand' --gradient_accumulation_steps 5")

# os.system(f"CUDA_VISIBLE_DEVICES=1 numactl --physcpubind=12-23 python train13.py --cfg './configs/ultramnist/grad_um_vit.py' --name 'grad_um_vit_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {50} --eval_every {50} --pretrained --logger_dir './vit_logger/um/grad' --gradient_accumulation_steps 5")
