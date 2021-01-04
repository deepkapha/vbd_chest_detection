"""
faster_rcnn.py - module for implementing Faster-RCNN
"""

# import dependencies
import torch
import torchvision

def FasterRCNN(pretrained = True):
	return torchvision.models.detection.fastercnn_resnet50_fpn(pretrained = pretrained)
