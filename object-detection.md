## Object Detection 
This page contains articles explaining concepts and code behind object detection. Living doc, trying to keep a sense of dates so that older resources can fall out of date appropriately.
Will gather more here in the future, especially around newer model SOTA architectures.

- Feb 2019 - detailed documented use of TF obj detection API.
https://towardsdatascience.com/creating-your-own-object-detector-ad69dda69c85

- TF obj detection tutorial. Uses pre-trained model to detect objects in an image.
https://colab.research.google.com/github/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

- July 2018 article from Sara about mobile & TPUs: 
https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193

- Nov 2018 "Object Detection using TensorFlow and COCO Pre-Trained Models": https://mc.ai/object-detection-using-tensorflow-and-coco-pre-trained-models/

- TF docs on "Distributed Training on the Oxford-IIIT Pets Dataset on Google Cloud". 
Pretty useful for getting up and running, though as of early 2020 it was still using TF 1.X, tf.slim, and Py27, 
which made for some confusion: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md

- GCP object detection example (pretrained only): https://github.com/GoogleCloudPlatform/tensorflow-object-detection-example

- TF docs on using a custom dataset: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md

- missinglink.ai resource, not sure how long this will stay up since they're shutting down. May need to look at a mirror: https://missinglink.ai/guides/tensorflow/tensorflow-image-recognition-object-detection-api-two-quick-tutorials/

- 5 part series from Dan Stang in Oct 2017:

  - https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-1-selecting-a-model-a02b6aabe39e
  - https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-2-converting-dataset-to-tfrecord-47f24be9248d
  - https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-3-creating-your-own-dataset-6369a4d30dfd
  - https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-4-training-the-model-68a9e5d5a333
  - https://medium.com/@WuStangDan/step-by-step-tensorflow-object-detection-api-tutorial-part-5-saving-and-deploying-a-model-8d51f56dbcf1

- Nov 2018: Faster R-CNN (object detection) implemented by Keras for custom data from Google’s Open Images Dataset V4: https://towardsdatascience.com/faster-r-cnn-object-detection-implemented-by-keras-for-custom-data-from-googles-open-images-125f62b9141a

- ML Mastery (May 2019) Obj Detection with Keras: https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

- fritz.ai 2019 guide to object detection: https://heartbeat.fritz.ai/a-2019-guide-to-object-detection-9509987954c3

- Review of Stanford vision ML course (CS231n): https://machinelearningmastery.com/stanford-convolutional-neural-networks-for-visual-recognition-course-review/

- cs231n: Convolutional Neural Networks for Visual Recognition (Fei-Fei Li): http://cs231n.stanford.edu/index.html

- cs230: Deep learning (Andrew Ng): https://cs230.stanford.edu/

- Github topic: Object Detection: https://github.com/topics/object-detection

- Demos, articles, code, etc (updated Nov 2019) : https://github.com/amusi/awesome-object-detection

  - based on https://handong1587.github.io/deep_learning/2015/10/09/object-detection.html
  
- an experimental Tensorflow implementation of Faster RCNN: https://github.com/smallcorgi/Faster-RCNN_TF

  - an experimental Tensor Flow implementation of Faster RCNN (TFFRCNN): https://github.com/CharlesShang/TFFRCNN
  
- TF model zoo for obj detection: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

- Jun 2019 Creating a Object Detection model from scratch using Keras: https://datascience.stackexchange.com/questions/53114/creating-a-object-detection-model-from-scratch-using-keras

- Aug 2019 Nvidia Jetson Nano: Custom Object Detection from scratch using Tensorflow and OpenCV: https://medium.com/swlh/nvidia-jetson-nano-custom-object-detection-from-scratch-using-tensorflow-and-opencv-113fe4dba134

- Custom Object Detection using TensorFlow from Scratch
TensorFlow Object Detection Training on Custom Dataset: https://towardsdatascience.com/custom-object-detection-using-tensorflow-from-scratch-e61da2e10087

- Oct 2019 YOLOv3 in Keras: https://machinelearningmastery.com/how-to-perform-object-detection-with-yolov3-in-keras/

- May 2019: How to Train an Object Detection Model with Keras. (Mask R-CNN model) https://machinelearningmastery.com/how-to-train-an-object-detection-model-with-keras/

- keras YOLOv3: https://github.com/experiencor/keras-yolo3

- YOLO homepage: https://pjreddie.com/darknet/yolo/

- Guide to Object Detection using Deep Learning: Faster R-CNN,YOLO,SSD (no code, just ideas) https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/

- TF docs: Transfer learning with a pretrained ConvNet: https://www.tensorflow.org/tutorials/images/transfer_learning

- Lambda labs: classify images with TF: https://lambdalabs.com/blog/how-to-classify-images-with-tensorflow-a-step-by-step-tutorial/

  - Transfer Learning with TensorFlow Tutorial: Image Classification https://lambdalabs.com/blog/transfer-learning-with-tensorflow-tutorial-image-classification-example/

  - Oct 2018: image segmentation: https://lambdalabs.com/blog/image-segmentation-i-beginners-demo-2/

  - Sept 2018: hyperparameter tuning: https://lambdalabs.com/blog/image-classfication-i-beginners-demo-2/

- GCP 'Solution': Creating an Object Detection Application Using TensorFlow. https://cloud.google.com/solutions/creating-object-detection-application-tensorflow
  - Note: This comes with a basic web frontend
