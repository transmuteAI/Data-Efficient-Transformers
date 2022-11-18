import os

# os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {10} --eval_every {6} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {20} --eval_every {6} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {50} --eval_every {6} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3 --train_batch_size 128 --learning_rate 0.0075")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {25} --select_every {10} --eval_every {5} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {25} --select_every {20} --eval_every {5} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {25} --select_every {50} --eval_every {5} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.3} --num_steps {1470} --warmup_steps {70} --select_every {10} --eval_every {14} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.3} --num_steps {1470} --warmup_steps {70} --select_every {20} --eval_every {14} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.3} --num_steps {1470} --warmup_steps {70} --select_every {50} --eval_every {14} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {10} --eval_every {23} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {20} --eval_every {23} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.5} --num_steps {2415} --warmup_steps {115} --select_every {50} --eval_every {23} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {10} --eval_every {36} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {20} --eval_every {36} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=4 numactl --physcpubind=48-59 python train13.py --cfg './configs/ultramnist/glis/glis_um_mobv3.py' --name 'glis_um_mobv3_pretrained' --fraction {0.8} --num_steps {3780} --warmup_steps {180} --select_every {50} --eval_every {36} --pretrained --logger_dir './mobv3_logger/um/glis' --gradient_accumulation_steps 3")
