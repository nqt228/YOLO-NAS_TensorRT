# YOLO-NAS TensorRT
Train and inference custom dataset on Jetson using TensorRT
## Prepare the environment
Install [`super-gradients`](https://github.com/Deci-AI/super-gradients) package for ONNX export or TensorRT API building.

   ``` shell
   pip install super-gradients
   ```
   ``` shell
   git clone https://github.com/nqt228/YOLO-NAS_TensorRT.git
   cd YOLO-NAS_TensorRT
   ```
## Train on custom dataset
#### Create a directory for custom dataset with yolo format:
``` shell
├── dataset
├───── train
│       ├── images
│       └── labels
└───── valid
        ├── images
        └── labels
```
#### Fill in your `'classes'` and `'data_dir'` in `train.py`:
```python
dataset_params = {
    'data_dir':'custom-dataset', #dataset directory
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
```
#### Train
``` shell
python3 train.py 
```
## Export ONNX with NMS
``` shell
$ python3 export_yolonas.py 
--arch yolo_nas_m /                                                                    
--ckpt yolo-nas.pth /
--out yolo-nas.onnx /                                                 
--classes 1 /
--size 640 /
--iou_thres 0.45 /
--score_thres 0.5 /
--topk 100 /
--batch 1
```
NOTE: Change `yolo-nas.pth` to the location of your model.   
## Export Engine by Trtexec Tools
On your target system:

``` shell
/usr/src/tensorrt/bin/trtexec \
--onnx=yolo-nas.onnx \
--saveEngine=yolo-nas.engine \
```
#### Inference
Build
``` shell
mkir build && cd build
cmake ..
make
```
Inference
``` shell
./yolonas <engine model> <test image>
```
Description of arguments

- `engine model`: The Engine model (yolo-nas.engine).
- `test image` : The image path.

The result is stored in the same directory name `result.jpg`

