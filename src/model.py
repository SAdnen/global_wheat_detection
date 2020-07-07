import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN, MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator
import torch.nn as nn


class Resnet50_fpn(nn.Module):
    def __init__(self):
        super(Resnet50_fpn, self).__init__()

        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained_backbone=True,
                                                                          rpn_pre_nms_top_n_train=1000,
                                                                          rpn_post_nms_top_n_train=1000,
                                                                          rpn_fg_iou_thresh=0.6,
                                                                          rpn_anchor_generator=rpn_anchor_generator,
                                                                          box_detections_per_img=120,
                                                                          )
        num_classes = 2


        inf_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(inf_features, num_classes)

    def forward(self, x, y=None):
        if y is not None:
            loss_dict = self.model(x, y)
            return loss_dict
        # print(loss_dict)
        else:
            predictions = self.model(x)
            return predictions

class Backbone(nn.Module):
    def __init__(self, ref):
        super(Backbone, self).__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', ref, pretrained=True).features
        self.out_channels = 1024

    def forward(self, x):
        return self.model(x)


class Fasterrcnn_densenet(nn.Module):
    def __init__(self, ref):
        super(Fasterrcnn_densenet, self).__init__()
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(
            anchor_sizes, aspect_ratios
        )

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0'],
            output_size=7,
            sampling_ratio=2)

        backbone = Backbone(ref)
        self.model = FasterRCNN(backbone=backbone,
                                num_classes=2,
                                box_roi_pool=box_roi_pool,
                                rpn_pre_nms_top_n_train=1000,
                                rpn_post_nms_top_n_train=1000,
                                rpn_fg_iou_thresh=0.6,
                                rpn_anchor_generator=rpn_anchor_generator,
                                box_detections_per_img=120,
                                )

    def forward(self, x, y=None):
        if y is not None:
            loss_dict = self.model(x, y)
            return loss_dict
        # print(loss_dict)
        else:
            predictions = self.model(x)
            return predictions



class Models(object):
    networks = {"resnet50_fpn": Resnet50_fpn(),
                "densenet121": Fasterrcnn_densenet('densenet121')}

    def get_model(self, reference):
        available_models = ", ".join(self.networks.keys())
        assert (reference in self.networks.keys()), "Please choose one of: [{}].".format(available_models)
        return self.networks[reference]


def load_checkpoint(model, checkpoint_path='global_wheat_fasterrcnn_epoch_5.pth'):
    # to be moved to engine/classifier
    device = "cuda" if torch.cuda.is_available() else torch.device("cpu")
    # checkpoint_path = path.parent/"input/checkpoint2/global_wheat_fasterrcnn_epoch_5.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model

if __name__ == "__main__":
    reference1 = "resnet"
    reference2 = "resnet50_fpn"
    models = Models()
    model1 = models.get_model(reference1)
    model2 = models.get_model(reference2)
    print("done")