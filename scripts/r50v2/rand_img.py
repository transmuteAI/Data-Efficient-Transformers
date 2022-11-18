import os

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {10} --eval_every {2} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {20} --eval_every {2} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.01} --num_steps {210} --warmup_steps {10} --select_every {50} --eval_every {2} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.05} --num_steps {945} --warmup_steps {45} --select_every {10} --eval_every {9} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.05} --num_steps {945} --warmup_steps {45} --select_every {20} --eval_every {9} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.05} --num_steps {945} --warmup_steps {45} --select_every {50} --eval_every {9} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.1} --num_steps {1890} --warmup_steps {90} --select_every {10} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.1} --num_steps {1890} --warmup_steps {90} --select_every {20} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.1} --num_steps {1890} --warmup_steps {90} --select_every {50} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.3} --num_steps {5565} --warmup_steps {265} --select_every {10} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.3} --num_steps {5565} --warmup_steps {265} --select_every {20} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.3} --num_steps {5565} --warmup_steps {265} --select_every {50} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.5} --num_steps {9240} --warmup_steps {440} --select_every {10} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.5} --num_steps {9240} --warmup_steps {440} --select_every {20} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.5} --num_steps {9240} --warmup_steps {440} --select_every {50} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.8} --num_steps {14805} --warmup_steps {705} --select_every {10} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.8} --num_steps {14805} --warmup_steps {705} --select_every {20} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")

os.system(f"CUDA_VISIBLE_DEVICES=2 numactl --physcpubind=12-23 python train13.py --cfg './configs/T-IMGNET/rand_img_r50v2.py' --name 'rand_img_r50v2_pretrained' --fraction {0.8} --num_steps {14805} --warmup_steps {705} --select_every {50} --pretrained --logger_dir './r50v2_logger/IMGNET/rand'")



