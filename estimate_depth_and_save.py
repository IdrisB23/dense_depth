import os
import glob
import argparse
import matplotlib
from pathlib import Path
import cv2

import sys, os
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from utility_functions import create_output_dir_if_req, clean_output_dir

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from tensorflow.keras.layers import Layer, InputSpec
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt


BASE_DIR = Path('..')
# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')

parser.add_argument('--orig_frames_path', default='sampled_frames/orig', 
                    type=str, help='Path to the extracted frames directory')
parser.add_argument('--output_path', default='sampled_frames/depth',
                    type=str, help='Path to where the processed frames should be saved')
parser.add_argument('--pretrained_model_path', default='pre-trained_models/nyu.h5', 
                    type=str, help='Pre-trained model path')

args = parser.parse_args()


def process_frames_via_denseDepth(input_frames_path: Path, output_path: Path, model_path: str):
    # Custom object needed for inference and training
    custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}
    print('Loading model...')
    # Load model into GPU / CPU
    model = load_model(args.pretrained_model_path, custom_objects=custom_objects, compile=False)
    print('\nModel loaded from ({0}).'.format(model_path))

    # Input images
    input_frames_path = input_frames_path.as_posix()
    inputs = load_images(glob.glob(input_frames_path + '/*.png'))
    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))

    # Compute results
    outputs = predict(model, inputs)

    #matplotlib problem on ubuntu terminal fix
    #matplotlib.use('TkAgg')
    for i in range(1, outputs.shape[0] + 1):
        # print(i)
        depth_map = outputs[i-1, ...]
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        # print(normalized_depth_map)
        # print('---/---')
        output_name = str(output_path / '{:03d}.png'.format(i))
        cv2.imwrite(output_name, normalized_depth_map)


if __name__ == '__main__':
    orig_frames_path = BASE_DIR / args.orig_frames_path
    output_path = BASE_DIR / args.output_path
    pretrained_model_path = args.pretrained_model_path
    create_output_dir_if_req(output_path)
    clean_output_dir(output_path)
    process_frames_via_denseDepth(orig_frames_path, 
                             output_path, 
                             pretrained_model_path)