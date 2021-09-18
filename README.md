# Detection of Skin Defects in Mangoes Using Superpixels and SVM

This program uses Superpixel segmentation and Support Vector Machines to detect skin defects in mangoes. A raw image of a mango taken from a camera phone is first enhanced by using image processing techniques and background removal. Superpixel segmentation is then performed to determine mango parts. Features are then extracted from the superpixels to be classified using Support Vector Machines (SVM). Lastly, assessment of “good” and “bad” mangoes is done by getting the ratio of defects with respect to the entire mango surface.

<h2> Dataset </h2>

Since there are no previously available dataset for fruits that can be used on computer vision processing (when this program was written: 2015), dataset was built by taking mango images using phone camera. The images collected consists of 79 raw images directly from the phone camera, 30 of which are considered as bad mangoes, while 49 are good mangoes. The dataset is then divided into two sets, 30 for training, and 49 for testing.

Aside from labelling if a mango is good or bad, the bad/rotten parts of the mango (in terms of superpixels) were also manually labelled. 

<h2> Pre-processing </h2>

One of the important aspects in machine learning is to get the best features that describe the dataset. Pre-processing improves the quality of the dataset by removing noise or unwanted features, and making good features more distinct. In computer vision, image enhancement techniques are used before the actual processing such as when performing feature detection and extraction.

The first pre-processing step done on the dataset images is enhancing the color. There are two basic aspects of color used to improve the images: saturation and contrast. Saturation is the intensity or colorfulness of an image. Increasing the saturation values of a muted image gives us more distinct color values. Contrast, on the other hand, is the difference in luminance that makes an object distinguishable. Combination of contrast and brightness gives us a clearer and better image. For the saturation processing, images were converted from RGB to HSV (or Hue-Saturation-Value) color space.
 
Another pre-processing step performed is background removal. This helps improve accuracy and efficiency of the algorithm since this made the background easier to identify (during classification step). To remove the background, edge detection using Canny algorithm was used. Since Canny accepts 1-channel image, only the S-channel or the saturation channel was used from the converted HSV image. This is used over the other channels as the saturation channel’s contour is well defined when displayed compared to hue and value as these latter two don’t have large variance given this particular dataset. Canny detected the shape of the mango, however, it also included some unnecessary artifacts inside the mango and some lines in background. Additionally, the contour of the mango is not closed looped, meaning, that there are gaps and breaks in the edge line.

To help close this loop, morphological closing was used in the edge mask returned by Canny. Morphological closing is an image enhancement technique obtained by dilation followed by erosion. Dilation and erosion are the main processes in morphological transformation. In dilation, the value of the output pixel is the maximum value of all the pixels in the input pixel's neighborhood, while it’s the minimum in erosion. Morphological closing can be used to remove dark regions near bright regions. Since in the Canny output, the edge is colored white, gaps between the edge’s pixels are closed using the morphological closing technique.

At this point, we now have a closed edge defining the contour of the mango. Next step is to fill up the inside of the mango with white. OpenCV findContour and drawContours functions were used. This binary image serves as the mask defining the foreground.

Finally, morphological opening was used. In this type, erosion was first done before dilation. This is useful for removing small objects that are bright over dark background. This was used to remove unwanted artifacts such as edges caused by a line in the background.

Sample raw and pre-processed image:

<img src="https://user-images.githubusercontent.com/90839613/133888589-8d8dd553-48e1-4f8c-b1a9-536fd1b547a9.png" width=500>

<h2>Segmentation</h2>

After pre-processing, the image is ready for segmentation. Image segmentation is the process of dividing an image into distinct regions, grouping together pixels that share certain characteristics. In this problem, the goal is to divide the image of a mango into three parts: the background, the “bad” parts of the mango (or the defect), and the “good” parts of the mango.

There are several image segmentation techniques. The most common methods are thresholding, clustering, and graph cut methods. There is also a technique that is getting popular recently called superpixelization. Unlike the usual segmentation algorithm, the aim of superpixel algorithms is to over-segment the image. This is the technique used in the program.

<h2>Superpixels and SEEDS</h2>

The superpixel algorithm used in this program is called SEEDS: Superpixels Extracted via Energy-Driven Sampling. SEEDS is efficient over other algorithms since typically, the other state-of-the-arts use sophisticated methods that progressively build the superpixels, i.e. by adding cuts or growing superpixels. This results to computationally expensive processing. SEEDS, on the other hand, is only based on a simple hill-climbing optimization.

To initialize the algorithm, there are parameters that need to be defined to acquire the preferred result. One of which is the number of block levels, which defines how many block should be used in each iteration. Another is smoothness value, which defines if the algorithm with try to stabilize the shape of the superpixel. The number of desired superpixels is also to be inputted. Several experiments were done on different parameter values to get the best segmentation results.

A sample input image and superpixelized version is displayed below:

<img src="https://user-images.githubusercontent.com/90839613/133888882-b5fe4cb4-700e-4257-aa5d-b742005e4a5e.png" width=500>

<h2> Feature Extraction </h2>

After getting the superpixel segments of the image, next step is getting the feature set that will be used in predicting the classification of each superpixel.

The values used are from RGB histogram of each superpixels, computed by counting the number of pixels within its superpixel group that has a specific R, G, and B value. The values in each channel ranges from 0 to 255, but 16 histogram bins were used, which means the 0 to 255 is divided into 16 sets. This resulted to 16x3 or 48 dimensions per superpixel.

After getting the RGB histogram, the values were normalized to range from 0 to 1. Normalization is needed since the value of the histogram is proportional to the number of pixels a certain superpixel has covered. Since the superpixels have different sizes, the bigger superpixel will overwhelm the smaller superpixels if not normalized.

<h2> Superpixel Level Classification using SVM </h2>

SVM was used to classify if each superpixel is a background, a defect, or a good part of the mango, label being 0, -1 and +1 respectively. Since these are 3-class case, one-vs-all approach was used. The kernel used is radial basis function or RBF.

To estimate the best parameters aside from the kernel type, grid search was done. Grid search is a methodology that estimates for best model by running the SVM using all possible parameter combination from the ranges specified by the user.

In addition, 5-fold cross validation was used in the grid search parameter estimation, and in getting the training accuracy. Libsvm was used to implement this section, and the results are below.

<h2> Mango Level Classification </h2>

Once the superpixel level classification is completed, the labeled images are now checked. Pixels under the positive class (good mango part) and negative class (bad mango part) are retrieved. Computation for the percentage of defects are performed, using the following formula:
<i>No. of pixels labeled as defects / No. of pixels labeled as mango (defect + normal skin)</i>

If the percentage of skin defects is greater than 0.5%, it is tagged as a bad mango.

<h2> Results </h2>

The accuracy of superpixel level and mango level results are shown below. As can be seen, the superpixel level accuracies are high, and the mango level accuracy is also good, provided that there are only a few number of images. There are 4 mangoes that failed out of the 49 mangoes in the test set.

<table>
  <tr>
    <th>Description</th>
    <th>Dataset</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <th>Superpixel Level: 5-fold Cross-Validation Accuracy</th>
    <td>Training set: ~5k superpixels from 30 mangoes</td>
    <td>98.2626 %</td>
  </tr>
  <tr>
    <th>Superpixel Level: Prediction Accuracy</th>
    <td>Test set: ~8k superpixels from 49 mangoes</td>
    <td>98.4386%</td>
  </tr>
  <tr>
    <th>Mango Level: Prediction Accuracy</th>
    <td>Test set: 49 mangoes</td>
    <td>91.8367%</td>
  </tr>
</table>

Following is the confusion matrix of the testing dataset. The precision and recall are both at 93.93%. It is good that the recall percentage is still high as we don’t want damaged mango being included in the good stack.

<table>
  <tr>
    <th></th>
    <td colspan=2>Actual Class</td>
  </tr>
  <tr>
    <th rowspan=2>Predicted Class</th>
    <td>TP = 31</td>
    <td>FN = 2</td>
  </tr>
  <tr>
    <td>FP = 2</td>
    <td>TN = 14</td>
  </tr>
  <tr>
    <th>Total</th>
    <td>TotP = 33</td>
    <td>TotN = 16</td>
  </tr>
  </table>
  
These are sample true positive mangoes. We can see that the labeled superpixels covered almost all of the mango parts.

<img src="https://user-images.githubusercontent.com/90839613/133889572-7765715a-c492-449f-b89b-eda444f4522e.png" width=500>

Below images show true negative examples. Here, the mango part where the skin defect is located and labeled correctly.

<img src="https://user-images.githubusercontent.com/90839613/133889634-1644e615-406f-457f-88da-0ea71653c151.png" width="500">

Next shows the two false positive cases. These are the positive cases (good mango) that are labeled negative. For the first example, there is no obvious reason why the black part was labeled as such by the algorithm, but in the second example, it might be either because of the stem or the shadow (darker area).

<img src="https://user-images.githubusercontent.com/90839613/133889695-0f9b073b-0159-4e7d-b5e8-25f4ea3bcfb6.png" width="500">

Finally, here are the two false negative cases. These are the mangoes with skin defects that are labeled as good. As can be observed, the defects were not that noticable as well with the naked eye.

<img src="https://user-images.githubusercontent.com/90839613/133889786-901be97d-af49-4f30-9a71-76b123a2c206.png" width="500">


