import argparse
import logging
import time

import numpy as np
import torch.utils.data

from hardware.device import get_device
from inference.post_process import post_process_output,post_process_output_trt
from utils.data import get_dataset
from utils.dataset_processing import evaluation, grasp
from utils.visualisation.plot import save_results

import tensorrt as trt
import inferencetest as inference_utils 
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate networks')

    # Network
    parser.add_argument('--onnx', type=str,default = "grcnn.onnx",
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--trt', type=str,default = "grcnn.trt",
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size for the network')

    # Dataset
    parser.add_argument('--dataset', type=str,default = "cornell",
                        help='Dataset Name ("cornell" or "jaquard")')
    parser.add_argument('--dataset-path', type=str,default='/media/xcy/TOSHIBA480/0.Datasets/cornell',
                        help='Path to dataset')
    parser.add_argument('--use-depth', type=int, default=0,
                        help='Use Depth image for evaluation (1/0)')
    parser.add_argument('--use-rgb', type=int, default=1,
                        help='Use RGB image for evaluation (1/0)')
    parser.add_argument('--augment', action='store_true',
                        help='Whether data augmentation should be applied')
    parser.add_argument('--split', type=float, default=0.9,
                        help='Fraction of data for training (remainder is validation)')
    parser.add_argument('--ds-shuffle', action='store_true', default=False,
                        help='Shuffle the dataset')
    parser.add_argument('--ds-rotate', type=float, default=0.0,
                        help='Shift the start point of the dataset to use a different test/train split')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Dataset workers')

    # Evaluation
    parser.add_argument('--n-grasps', type=int, default=1,
                        help='Number of grasps to consider per image')
    parser.add_argument('--iou-threshold', type=float, default=0.25,
                        help='Threshold for IOU matching')
    parser.add_argument('--iou-eval', type=bool,default=True,
                        help='Compute success based on IoU metric.')
    parser.add_argument('--jacquard-output', action='store_true',
                        help='Jacquard-dataset style output')

    # Misc.
    parser.add_argument('--vis', type=bool,default=False,
                        help='Visualise the network output')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='Force code to run in CPU mode')
    parser.add_argument('--random-seed', type=int, default=123,
                        help='Random seed for numpy')

    args = parser.parse_args()

    if args.jacquard_output and args.dataset != 'jacquard':
        raise ValueError('--jacquard-output can only be used with the --dataset jacquard option.')
    if args.jacquard_output and args.augment:
        raise ValueError('--jacquard-output can not be used with data augmentation.')

    return args


if __name__ == '__main__':
    args = parse_args()

    # Get the compute device
    device = get_device(args.force_cpu)

    # Load Dataset
    logging.info('Loading {} Dataset...'.format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)
    test_dataset = Dataset(args.dataset_path,
                           output_size=args.input_size,
                           ds_rotate=args.ds_rotate,
                           random_rotate=args.augment,
                           random_zoom=args.augment,
                           include_depth=args.use_depth,
                           include_rgb=args.use_rgb)

    indices = list(range(test_dataset.length))
    split = int(np.floor(args.split * test_dataset.length))
    if args.ds_shuffle:
        np.random.seed(args.random_seed)
        np.random.shuffle(indices)
    val_indices = indices[split:]
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices)
    logging.info('Validation size: {}'.format(len(val_indices)))

    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=args.num_workers,
        sampler=val_sampler
    )
    logging.info('Done')

    # 1. 网络构建
    # Precision command line argument -> TRT Engine datatype
    TRT_PRECISION_TO_DATATYPE = {
        16: trt.DataType.HALF,
        32: trt.DataType.FLOAT
    }
    # datatype: float 32
    trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[16]
    # batch size = 1
    max_batch_size = 1
    engine_file_path = "grcnn.trt"
    onnx_file_path = "grcnn.onnx"
    new_width, new_height = 224, 224
    output_shapes = [(1, new_height, new_width)]
    trt_inference_wrapper = inference_utils.TRTInference(
        engine_file_path, onnx_file_path,
        trt_engine_datatype, max_batch_size,
    )

    results = {'correct': 0, 'failed': 0}

    start_time = time.time()

    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            model_out = trt_inference_wrapper.infer(x, output_shapes, new_width, new_height)
            # print(model_out.size)
            pred_pos_out = model_out[0][0]
            pred_cos_out = model_out[1][0]
            pred_sin_out = model_out[2][0]
            pred_width_out = model_out[3][0]
            q_img, ang_img, width_img = post_process_output_trt(pred_pos_out, pred_cos_out, pred_sin_out, pred_width_out)

            if args.iou_eval:
                s = evaluation.calculate_iou_match(q_img, ang_img, test_data.dataset.get_gtbb(didx, rot, zoom),
                                                    no_grasps=args.n_grasps,
                                                    grasp_width=width_img,
                                                    threshold=args.iou_threshold
                                                    )
                if s:
                    results['correct'] += 1
                else:
                    results['failed'] += 1

            if args.vis:
                save_results(
                    rgb_img=test_data.dataset.get_rgb(didx, rot, zoom, normalise=False),
                    depth_img=test_data.dataset.get_depth(didx, rot, zoom),
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=args.n_grasps,
                    grasp_width_img=width_img
                )

    avg_time = (time.time() - start_time) / len(test_data)
    logging.info('Average evaluation time per image: {}ms'.format(avg_time * 1000))

    if args.iou_eval:
        logging.info('IOU Results: %d/%d = %f' % (results['correct'],
                                                    results['correct'] + results['failed'],
                                                    results['correct'] / (results['correct'] + results['failed'])))

