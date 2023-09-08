import sys
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val
from IPython.display import clear_output
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback


CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='custom', ckpt_root_dir=CHECKPOINT_DIR)

dataset_params = {
    'data_dir':'dataset', #dataset directory
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    # 'test_images_dir':'test/images',
    # 'test_labels_dir':'test/labels',

    'classes': ['car'], #Fill in your classess
    
    'transforms':[{'DetectionRandomAffine': {'degrees': 0,'translate':0.25,'scales':(0.5,1.5), 'shear': 0.0, 'target_size':'','filter_box_candidate': True,'wh_thr':2, 'area_thr':0.1, 'ar_thr':20}}, 
    {'DetectionHSV': {'prob': 0.5, 'hgain': 18, 'sgain':30, 'vgain':30}}, 
    {'DetectionHorizontalFlip': {'prob': 0.5}},
    {'DetectionMixup':{'input_dim':(640,640),'mixup_scale': (0.5,1.5),'prob':0.5, 'flip_prob':0.5 }},
    {'DetectionPadToSize':{'output_size': (640,640),'pad_value':114}},
    {'DetectionStandardize': {'max_value':255.}},
    'DetectionImagePermute',
    {'DetectionTargetsFormatTransform':{'input_dim':(640,640), 'output_format':'LABEL_CXCYWH'}}
    ]
}

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes'],
        'transforms': [
    {'DetectionRandomAffine': {'degrees': 0,'translate':0.25,'scales':(0.5,1.5), 'shear': 0.0, 'target_size':(640,640),'filter_box_candidates': True,'wh_thr':2, 'area_thr':0.1, 'ar_thr':20}}, 
    {'DetectionHSV': {'prob': 0.5, 'hgain': 18, 'sgain':30, 'vgain':30}}, 
    {'DetectionHorizontalFlip': {'prob': 0.5}},
    {'DetectionMixup':{'input_dim':(640,640),'mixup_scale': (0.5,1.5), 'prob':0.5, 'flip_prob':0.5 }},
    {'DetectionPadToSize':{'output_size': (640,640),'pad_value':114}},
    {'DetectionStandardize': {'max_value':255.}},
    'DetectionImagePermute',
    {'DetectionTargetsFormatTransform':{'input_dim':(640,640), 'output_format':'LABEL_CXCYWH'}}]
    },
    dataloader_params={
        'batch_size':8,
        'num_workers':2
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes'],
        'transforms': [{'DetectionPadToSize':{'output_size': (640,640),'pad_value':114}},{'DetectionStandardize': {'max_value':255.}},
        'DetectionImagePermute',{'DetectionTargetsFormatTransform':{'input_dim':(640,640), 'output_format':'LABEL_CXCYWH'}}]
    },
    dataloader_params={
        'batch_size':8,
        'num_workers':2
    }
)

# test_data = coco_detection_yolo_format_val(
#     dataset_params={
#         'data_dir': dataset_params['data_dir'],
#         'images_dir': dataset_params['test_images_dir'],
#         'labels_dir': dataset_params['test_labels_dir'],
#         'classes': dataset_params['classes']
#     },
#     dataloader_params={
#         'batch_size':2,
#         'num_workers':2
#     }
# )

clear_output()

train_data.dataset.transforms

train_data.dataset.plot()

model = models.get('yolo_nas_m',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
                   )



train_params = {
    # ENABLING SILENT MODE
    'silent_mode': False,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # launch_tensorboard: False # Whether to launch a TensorBoard process.
    # tensorboard_port: # port for tensorboard process
    # tb_files_user_prompt: False  # Asks User for Tensorboard Deletion Prompt
    # save_tensorboard_to_s3: False # whether to save tb to s3
    # ONLY TRAINING FOR 10 EPOCHS FOR THIS EXAMPLE NOTEBOOK
    # save_model: True # Whether to save the model checkpoints
    # ckpt_best_name: ckpt_best.pth
    "max_epochs": 100,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # NOTE: num_classes needs to be defined here
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # NOTE: num_classes needs to be defined here
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

trainer.train(model=model,
              training_params=train_params,
              train_loader=train_data,
              valid_loader=val_data)

