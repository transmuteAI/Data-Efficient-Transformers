import os

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {10} --eval_every {4} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 2")
# --train_batch_size 256 --learning_rate 0.015

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {20} --eval_every {4} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 2")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {50} --eval_every {4} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 2")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.1} --num_steps {1890} --warmup_steps {90} --select_every {10} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.1} --num_steps {1890} --warmup_steps {90} --select_every {20} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.1} --num_steps {1890} --warmup_steps {90} --select_every {50} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.3} --num_steps {5565} --warmup_steps {265} --select_every {10} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.3} --num_steps {5565} --warmup_steps {265} --select_every {20} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.3} --num_steps {5565} --warmup_steps {265} --select_every {50} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.5} --num_steps {9240} --warmup_steps {440} --select_every {10} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.5} --num_steps {9240} --warmup_steps {440} --select_every {20} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

# os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.5} --num_steps {9240} --warmup_steps {440} --select_every {50} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.8} --num_steps {14805} --warmup_steps {705} --select_every {10} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.8} --num_steps {14805} --warmup_steps {705} --select_every {20} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=24-35 python train13.py --cfg './configs/T-IMGNET/swin/craig_img.py' --name 'craig_img_swin_pretrained' --fraction {0.8} --num_steps {14805} --warmup_steps {705} --select_every {50} --pretrained --logger_dir './swin_logger/imgnet/craig' --gradient_accumulation_steps 3")
