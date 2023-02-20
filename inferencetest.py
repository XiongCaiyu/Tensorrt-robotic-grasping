import os
import sys
import time
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
 
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    class HostDeviceMem(object):
        def __init__(self, host_mem, device_mem):
            self.host = host_mem
            self.device = device_mem

        def __str__(self):
            return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

        def __repr__(self):
            return self.__str__()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))

    return inputs, outputs, bindings, stream

def load_engine(trt_path):
    # 反序列化引擎
    with open(trt_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


class TRTInference(object):
    """Manages TensorRT objects for model inference."""
 
    def __init__(self, trt_engine_path, onnx_model_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1):
        """Initializes TensorRT objects needed for model inference.
        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """
 
        # Initialize runtime needed for loading TensorRT engine from file
        # TRT engine placeholder
        self.trt_engine = None
 
        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))
        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = load_engine(
                trt_engine_path)
 
        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.trt_engine)
 
        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()
 
    def infer(self, rgb_img, output_shapes, new_width, new_height):
        """Infers model on given image.
        Args:
            image_path (str): image to run object detection model on
        """
        
        assert new_width > 0 and new_height > 0, "Scale is too small"
        # resize and transform to array
        # rgb_img = cv2.resize(rgb_img,(new_width,new_height))
        # print("scale_depth image shape:{}".format(rgb_img.shape))
        # scale_img = np.array(scale_img)
        # HWC to CHW
        scale_img = rgb_img
        # scale_img = scale_img.transpose((2, 0, 1))
        # 归一化
        if scale_img.max() > 1:
            scale_img = scale_img / 255
        # 扩增通道数
        # scale_img = np.expand_dims(scale_img, axis=0)
        # 将数据成块

        scale_img = np.array(scale_img, dtype=np.float32, order='C')
        print("scale_img = {}".format(scale_img))
        # Copy it into appropriate place into memory
        # (self.inputs was returned earlier by allocate_buffers())
        np.copyto(self.inputs[0].host, scale_img.ravel())
        # Output shapes expected by the post-processor
        # output_shapes = [(1, 11616, 4), (11616, 21)]
        # When infering on single image, we measure inference
        # time to output it to the user
        inference_start_time = time.time()
 
        # Fetch output from the model
        trt_outputs = do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)
        print("network output shape:{}".format(trt_outputs[0].shape))

        #pred_pos.reshape(1,224,224)
        # # Output inference time
        print("TensorRT inference time: {} ms".format(
             int(round((time.time() - inference_start_time) * 1000))))
        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will
        # give us flat arrays.
        pred_pos = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        trt_outputs = trt_outputs[1:4]

        pred_cos = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        trt_outputs = trt_outputs[1:3]

        pred_sin = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        trt_outputs =trt_outputs[1:2]

        pred_width = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        #outputs = pred_pos.reshape(1,224,224)
        # print("pred_pos value ={}".format(pred_pos))
        # print("pred_cos value ={}".format(pred_cos))
        # print("pred_sin value ={}".format(pred_sin))
        # print("pred_width value ={}".format(pred_width))
     #   print("list is{}".format(list(zip(trt_outputs,[(1,224,224),(1,224,224)]))))
        # print("pre_output_shape:{}".format(output_shapes))
        # And return results
        return pred_pos,pred_cos,pred_sin,pred_width
 
 
# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    # return [out.host for out in outputs]
   # print("outputs = {}".format(outputs))
    return [out1.host for out1 in outputs]