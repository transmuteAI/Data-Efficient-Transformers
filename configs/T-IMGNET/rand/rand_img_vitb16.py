# Learning setting
config = dict(
                setting="SL",
                name="imgnet",
                model_type="ViT-B_16",
                pretrained_dir="checkpoint/ViT-B_16.npz",
                output_dir="output",
                img_size=224,
                train_batch_size=512,
                eval_batch_size=64,
                eval_every=100,
                learning_rate=3e-2,
                weight_decay=0,
                num_steps=10000,
                decay_type="cosine",
                warmup_steps=500,
                max_grad_norm=1.0,
                local_rank=-1,
                seed=42,
                gradient_accumulation_steps=3,
                fp16=True,
                fp16_opt_level="O2",
                loss_scale=0,
                is_reg = False,
                dataset=dict(
                    name="imgnet",
                    datadir="../storage",
                    feature="dss",
                    type="image"),

              dataloader=dict(
                    shuffle=True,
                    batch_size=128,
                    pin_memory=True,
                    num_workers=8,
                              ),

              model=dict(
                    architecture='ViT-B_16', #vit_base_patch16_224
                    numclasses=200,
                    pretrained=True,
                        ),
              
              ckpt=dict(is_load=False,
                        is_save=True,
                        dir='results/',
                        save_every=20),
              
              loss=dict(type='CrossEntropyLoss',
                        use_sigmoid=False),

            #   optimizer=dict(type="sgd",
            #                  momentum=0.9,
            #                 #  lr=0.01,
            #                  weight_decay=5e-4,
            #                  nesterov=False),

            #   scheduler=dict(type="cosine_annealing",
            #                  T_max=100),

            dss_args=dict(type="Random",
                                fraction=0.1,
                                select_every=20,
                                kappa=0),

              train_args=dict(num_epochs=100,
                              device="cuda",
                              print_every=10,
                              results_dir='results/',
                              print_args=["val_loss", "val_acc", "tst_loss", "tst_acc", "time"],
                              return_args=[]
                              )
              )
