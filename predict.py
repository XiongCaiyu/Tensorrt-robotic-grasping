import tensorrt as trt
import numpy as np
import cv2
import inferencetest as inference_utils  # TRT/TF inference wrappers
from utils.data.camera_data import CameraData
from PIL import Image
import matplotlib.pyplot as plt
import torch
from inference.post_process import post_process_output,post_process_output_trt
from utils.visualisation.plot import plot_results, save_results

if __name__ == "__main__":

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
    
    # 2. 图像预处理
    rgb_path = "pcd0101r.png"
    depth_path = "pcd0101d.tiff"
    img_data = CameraData(include_depth=0, include_rgb=1)
    # Load image
    pic = Image.open(rgb_path, 'r')
    rgb = np.array(pic)
    pic = Image.open(depth_path, 'r')
    depth = np.expand_dims(np.array(pic), axis=2)
    x, depth_img, rgb_img = img_data.get_data(rgb=rgb, depth=depth)
    
    # print("x_old={}".format(x))
    # # inference
    model_out = trt_inference_wrapper.infer(x, output_shapes, new_width, new_height)
    # print(model_out.size)
    pred_pos_out = model_out[0][0]
    pred_cos_out = model_out[1][0]
    pred_sin_out = model_out[2][0]
    pred_width_out = model_out[3][0]
    print("pred_pos = {}".format(pred_pos_out))
    # 输出后处理
    print("the size of pred_pos_out output : {}".format(pred_pos_out.shape))
    # pred_pos_out = pred_pos_out.transpose((1, 2, 0))
    # pred_cos_out = pred_cos_out.transpose((1, 2, 0))
    # pred_sin_out = pred_sin_out.transpose((1, 2, 0))
    # pred_width_out = pred_width_out.transpose((1, 2, 0))

    q_img, ang_img, width_img = post_process_output_trt(pred_pos_out, pred_cos_out, pred_sin_out, pred_width_out)
    # 0/1像素值
    print(q_img.shape)
   # print("the size of width_img : {}".format(width_img.shape))
    fig = plt.figure(figsize=(10, 10))
   # print(rgb_img.shape)
    plot_results(fig=fig,
                    rgb_img=img_data.get_rgb(rgb, False),
                    grasp_q_img=q_img,
                    grasp_angle_img=ang_img,
                    no_grasps=1,
                    grasp_width_img=width_img)
    plt.pause(0)
    # save_results(
    #             rgb_img=cam_data.get_rgb(img, False),
    #             depth_img=np.squeeze(cam_data.get_depth(depth)),
    #             grasp_q_img=q_img,
    #             grasp_angle_img=ang_img,
    #             no_grasps=1,
    #             grasp_width_img=width_img
    #         )
    # output = output.astype(np.uint8)
    # result = cv2.resize(output, (img.shape[1], img.shape[0]))
    # cv2.imwrite("best_output_deconv.jpg", result)