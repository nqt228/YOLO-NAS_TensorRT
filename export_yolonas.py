from super_gradients.training import models
import torch
import torch.nn as nn
import onnx
import argparse


class TRT_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=0,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
        class_agnostic=1
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)
        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=0,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25,
                 class_agnostic=1):
        out = g.op("TRT::EfficientNMS_TRT",
                   boxes,
                   scores,
                   background_class_i=background_class,
                   box_coding_i=box_coding,
                   iou_threshold_f=iou_threshold,
                   max_output_boxes_i=max_output_boxes,
                   plugin_version_s=plugin_version,
                   class_agnostic_i=class_agnostic,
                   score_activation_i=score_activation,
                   score_threshold_f=score_threshold,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes

class ONNX_TRT(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.65, score_thres=0.5, max_wh=None ,device=None, n_classes=80):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 0,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.n_classes=n_classes

    def forward(self, x):
        boxes, confscores = x
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(boxes, confscores, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes




def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-NAS conversion')
    parser.add_argument('-a', '--arch', required=True, help='YOLO-NAS architecture')
    parser.add_argument('-c', '--ckpt', required=True, help='Checkpoint model name (required)')
    parser.add_argument('-m', '--out', required=True, help='output model (.onnx) file path (required)')
    parser.add_argument('-n', '--classes', type=int, default=1, help='Number of trained classes (default 80)')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=640, help='Inference size [H,W] (default [640])')
    parser.add_argument('--iou_thres', type=float, default=0.45, help='Iou threshold')
    parser.add_argument('--score_thres', type=float, default=0.5, help='Score threshold')
    parser.add_argument('--topk', type=int, default=100, help='Number of maximum detected objects')
    parser.add_argument('--batch', type=int, default=1, help='Implicit batch-size')
    args = parser.parse_args()
    return args


def main(args):
    arch = args.arch
    NO_CLASSES = args.classes
    batch_size = args.batch
    topk_all = args.topk
    input_shape = args.size
    input_shape = (3, args.size, args.size) 
    iou_thres= args.iou_thres
    score_thres= args.score_thres
    onnx_ouput = args.out
    

    NMS = ONNX_TRT(max_obj=topk_all, iou_thres=iou_thres, score_thres=score_thres, max_wh=None ,device=None, n_classes=NO_CLASSES)
    NMS.eval()

    net = models.get(arch, checkpoint_path=args.ckpt, num_classes=NO_CLASSES)
    net.eval()
    net.prep_model_for_conversion()
    onnx_export_kwargs = {
    'input_names' : ['images'],
    'output_names' : ["num_dets", "det_boxes", "det_scores", "det_classes"]
    }
    models.convert_to_onnx(model=net, input_shape=input_shape, out_path=onnx_ouput,torch_onnx_export_kwargs=onnx_export_kwargs,post_process=NMS)
    batch_size = 1
    topk_all = 100
    shapes = [batch_size, 1,
              batch_size, topk_all, 4,
              batch_size, topk_all,
              batch_size, topk_all]
    onnx_model = onnx.load(onnx_ouput)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    for i in onnx_model.graph.output:
        for j in i.type.tensor_type.shape.dim:
            j.dim_param = str(shapes.pop(0))

    onnx.save(onnx_model, onnx_ouput)


if __name__ == '__main__':
    args = parse_args()
    main(args)
