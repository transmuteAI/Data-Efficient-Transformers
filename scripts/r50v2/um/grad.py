import os

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {10} --eval_every {6} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {20} --eval_every {6} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {50} --eval_every {6} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {25} --select_every {10} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {25} --select_every {20} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {25} --select_every {50} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.3} --num_steps {1470} --warmup_steps {70} --select_every {10} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.3} --num_steps {1470} --warmup_steps {70} --select_every {20} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.3} --num_steps {1470} --warmup_steps {70} --select_every {50} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {10} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {20} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/ultramnist/grad_um_r50v2.py' --name 'grad_um_r50v2_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {50} --pretrained --logger_dir './r50v2_logger/um/grad' --gradient_accumulation_steps 2")


