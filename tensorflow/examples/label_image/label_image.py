# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse

import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import os

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  print("type of file reader",type(file_reader))
  if file_name.endswith(".png"):
    image_reader = tf.io.decode_png(file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(tf.io.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.io.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.io.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.compat.v1.Session()
  return sess.run(normalized)

def load_labels(label_file):
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  return [l.rstrip() for l in proto_as_ascii_lines]


if __name__ == "__main__":
  file_name = "tensorflow/examples/label_image/data/grace_hopper.jpg"
  video_file_name = "tensorflow/examples/label_image/data/dog_short.mp4"
  output_video_name = "tensorflow/examples/label_image/data/output_video.avi"
  output_image_name = "tensorflow/examples/label_image/data/output_image.jpg"

  model_file = \
    "tensorflow/examples/label_image/data/inception_v3_2016_08_28_frozen.pb"
  label_file = "tensorflow/examples/label_image/data/imagenet_slim_labels.txt"
  input_height = 299
  input_width = 299
  input_mean = 0
  input_std = 255
  input_layer = "input"
  output_layer = "InceptionV3/Predictions/Reshape_1"

  parser = argparse.ArgumentParser()
  parser.add_argument("--image", help="image to be processed")
  parser.add_argument("--video", type = str , help="video to be processed")
  parser.add_argument("--output_video", help="path of output video location")
  parser.add_argument("--output_image", help="path of output image location")
  parser.add_argument("--graph", help="graph/model to be executed")
  parser.add_argument("--labels", help="name of file containing labels")
  parser.add_argument("--input_height", type=int, help="input height")
  parser.add_argument("--input_width", type=int, help="input width")
  parser.add_argument("--input_mean", type=int, help="input mean")
  parser.add_argument("--input_std", type=int, help="input std")
  parser.add_argument("--input_layer", help="name of input layer")
  parser.add_argument("--output_layer", help="name of output layer")
  args = parser.parse_args()

  if args.graph:
    model_file = args.graph
  if args.image:
    file_name = args.image
  if args.labels:
    label_file = args.labels
  if args.input_height:
    input_height = args.input_height
  if args.input_width:
    input_width = args.input_width
  if args.input_mean:
    input_mean = args.input_mean
  if args.input_std:
    input_std = args.input_std
  if args.input_layer:
    input_layer = args.input_layer
  if args.output_layer:
    output_layer = args.output_layer
  #Adding video argument
  if args.video:
    video_file_name = args.video
  #Adding output video argument
  if args.output_video:
    output_video_name = args.output_video
  #Adding output image argument
  if args.output_image:
    output_image_name = args.output_image


  graph = load_graph(model_file)

  # If only image parameter is given then it will process for image.
  if args.image and (not args.video) :
    t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.compat.v1.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0], {
          input_operation.outputs[0]: t
      })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    label , conf = labels[top_k[0]] , results[top_k[0]]*100
    print(label,conf)
    frame = cv2.imread(file_name)
    
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX 
    # org
    org = (10,input_height)
    # fontScale
    fontScale = 1
    # Blue color in BGR
    if conf > 80 :
      color = (0,255, 0)
    else:
      color = (0,0,255)

    # Line thickness of 2 px
    thickness = 2

    frame = cv2.putText(frame,label, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)

    cv2.imwrite(output_image_name,frame)
    print('Wrote the output image')

  else:

    cap = cv2.VideoCapture(video_file_name)
    ret, frame = cap.read()
    video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc(*'MJPG'), 20, (frame.shape[1], frame.shape[0]))
    ret = True
    
    while ret:
      ret, frame = cap.read()
      if not ret:
        break
      
      cv2.imwrite('intermediate_frame.jpg',frame)
      file_name = 'intermediate_frame.jpg'
      t = read_tensor_from_image_file(
        file_name,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)
      
      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      input_operation = graph.get_operation_by_name(input_name)
      output_operation = graph.get_operation_by_name(output_name)

      with tf.compat.v1.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
      results = np.squeeze(results)

      top_k = results.argsort()[-5:][::-1]
      labels = load_labels(label_file)

      label , conf = labels[top_k[0]] , results[top_k[0]]*100
      print("Frame classification result",label,conf)
      frame = cv2.imread(file_name)
      
      # font
      font = cv2.FONT_HERSHEY_SIMPLEX 
      # org
      org = (10,30)
      # fontScale
      fontScale = 1
      # Blue color in BGR
      if conf > 80 :
        color = (0,255, 0)
      else:
        color = (0,0,255)

      # Line thickness of 2 px
      thickness = 2
      # Using cv2.putText() method
      frame = cv2.putText(frame,f"{label} {conf}", org, font, 
                         fontScale, color, thickness, cv2.LINE_AA)
      video.write(frame)
      
    print('Video saved')
    os.remove(file_name)
