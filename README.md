# TensorFlow Lite Object Detection in Python

This code snipset is heavily based on <b><a href="https://www.tensorflow.org/lite/examples/object_detection/overview#example_applications_and_guides">TensorFlow Lite Object Detection</a></b><br>
The detection model can be downloaded from above link.<br>
For the realtime implementation on Android look into the <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">Android Object Detection Example</a><br>
Follow the <a href="https://github.com/joonb14/TFLiteDetection/blob/main/object%20detection.ipynb">object detection.ipynb</a> to get information about how to use the TFLite model in your Python environment.<br>

### Details
The <b>ssd_mobilenet_v1_1_metadata_1.tflite</b> file's input takes normalized 300x300x3 shape image. And the output is composed of 4 different outputs. The 1st output contains the bounding box locations, 2nd output contains the label number of the predicted class, 3rd output contains the probabilty of the image belongs to the class, 4th output contains number of detected objects(maximum 10). The specific labels of the classes are stored in the <b>labelmap.txt</b> file.<br>
I found the <b>labelmap.txt</b> in the <a href="https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android">Android Object Detection Example</a> repository in below directory.<br>
```
TFLite_examples/lite/examples/object_detection/android/app/src/main/assets
```
For model inference, we need to load, resize, typecast the image.<br>
The mobileNet model uses uint8 format so typecast numpy array to uint8.<br>
<img src="https://user-images.githubusercontent.com/30307587/110290738-75e93200-802e-11eb-92af-c3884b19f3bb.png" width=800px/><br>
Then if you follow the correct instruction provided by Google in <a href="https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_python">load_and_run_a_model_in_python</a>, you would get output in below shape<br>
<img src="https://user-images.githubusercontent.com/30307587/110290885-9f09c280-802e-11eb-9bec-52fec36e9beb.png" width=600px/><br>
Now we need to process this output to use it for object detection<br>

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

label_names = [line.rstrip('\n') for line in open("labelmap.txt")]
label_names = np.array(label_names)
numDetectionsOutput = int(np.minimum(numDetections[0],10))

for i in range(numDetectionsOutput):
    # Create figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(res_im)

    # Create a Rectangle patch
    inputSize = 300
    left = outputLocations[0][i][1] * inputSize
    top = outputLocations[0][i][0] * inputSize
    right = outputLocations[0][i][3] * inputSize
    bottom = outputLocations[0][i][2] * inputSize
    class_name = label_names[int(outputClasses[0][i])]
    print("Output class: "+class_name+" | Confidence: "+ str(outputScores[0][i]))
    rect = patches.Rectangle((left, bottom), right-left, top-bottom, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()
```
<img src="https://user-images.githubusercontent.com/30307587/110291065-d37d7e80-802e-11eb-8935-a5354a20df59.png" width=400px/><br>
I believe you can modify the rest of the code as you want by yourself.<br>
Thank you!<br>
