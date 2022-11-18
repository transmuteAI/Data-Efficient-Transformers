import os

# 2966

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {10} --eval_every {2} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 16 --learning_rate 9.375e-3")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {20} --eval_every {2} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 16 --learning_rate 9.375e-3")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {50} --eval_every {2} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 16 --learning_rate 9.375e-3")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {15} --select_every {10} --eval_every {3} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {15} --select_every {20} --eval_every {3} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.1} --num_steps {525} --warmup_steps {15} --select_every {50} --eval_every {3} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 128 --learning_rate 0.0075")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.3} --num_steps {210} --warmup_steps {10} --select_every {10} --eval_every {2} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 512")

os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python aptos.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.3} --num_steps {210} --warmup_steps {10} --select_every {20} --eval_every {10} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 512")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.3} --num_steps {210} --warmup_steps {10} --select_every {50} --eval_every {2} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1 --train_batch_size 512")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.5} --num_steps {315} --warmup_steps {15} --select_every {10} --eval_every {3} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.5} --num_steps {315} --warmup_steps {15} --select_every {20} --eval_every {3} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.5} --num_steps {315} --warmup_steps {15} --select_every {50} --eval_every {3} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.8} --num_steps {525} --warmup_steps {25} --select_every {10} --eval_every {50} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.8} --num_steps {525} --warmup_steps {25} --select_every {20} --eval_every {50} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1")

# os.system(f"CUDA_VISIBLE_DEVICES=3 numactl --physcpubind=36-47 python train13.py --cfg './configs/aptos/rn50/grad_ap_r50v2.py' --name 'grad_ap_r50v2_pretrained' --fraction {0.8} --num_steps {525} --warmup_steps {25} --select_every {50} --eval_every {50} --pretrained --logger_dir './r50v2_logger/ap/grad' --gradient_accumulation_steps 1")


