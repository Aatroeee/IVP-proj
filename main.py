import numpy as np
import cv2
import open3d as o3d
import os
import sys
import json
import argparse
import torch
from scipy.spatial import KDTree
import copy

from read_depth_and_build_pcd import read_depth_image, read_color_image, build_point_cloud_from_depth
from cam_settings import cam_series, keyframe_list, camera_set