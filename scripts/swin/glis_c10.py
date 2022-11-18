import os

# os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.01} --num_steps {105} --warmup_steps {5} --select_every {10} --eval_every {2} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

# os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.01} --num_steps {105} --warmup_steps {5} --select_every {20} --eval_every {2} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

# os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.01} --num_steps {105} --warmup_steps {5} --select_every {50} --eval_every {2} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.1} --num_steps {1050} --warmup_steps {50} --select_every {10} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.1} --num_steps {1050} --warmup_steps {50} --select_every {20} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.1} --num_steps {1050} --warmup_steps {50} --select_every {50} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.3} --num_steps {3045} --warmup_steps {145} --select_every {10} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.3} --num_steps {3045} --warmup_steps {145} --select_every {20} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.3} --num_steps {3045} --warmup_steps {145} --select_every {50} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.5} --num_steps {5145} --warmup_steps {245} --select_every {10} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.5} --num_steps {5145} --warmup_steps {245} --select_every {20} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.5} --num_steps {5145} --warmup_steps {245} --select_every {50} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.8} --num_steps {8190} --warmup_steps {390} --select_every {10} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.8} --num_steps {8190} --warmup_steps {390} --select_every {20} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")

os.system(f"CUDA_VISIBLE_DEVICES=5 numactl --physcpubind=60-71 python train13.py --cfg './configs/CIFAR10/swin/glis_c10_swin.py' --name 'glis_c10_swin_pretrained' --fraction {0.8} --num_steps {8190} --warmup_steps {390} --select_every {50} --pretrained --logger_dir './swin_logger/c10/glis' --gradient_accumulation_steps 6")
