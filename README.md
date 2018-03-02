# Image Processing

Adjust luminance using linear scaling and histogram equalization.

### Installing

Environment: python3
Install Package: pip install opencv-python

## Program description
1. Linear Scaling Method:
- First find the min L and max L at the specified window(H1, W1) to (H2, W2).
Then transfer the RGB color space to luv and apply linear scaling to whole image. 
Finally transfer back to RGB color space.
- During the transformation of XYZ to LUV, I add the condition to D == 0 since the dominator of the equation cannot be 0. Apply l'hospital law we have u’ = 4 and v’ = 9 if d == 0.
- During the transformation of LUV to XYZ, I add the condition to L == 0 since the dominator of the equation cannot be 0. Apply l'hospital law we have u = (13*uw)/13 and v = (13*vw)/13 if L == 0.
- Since we apply linear scaling to L, the RGB value transfer from XYZ might out of the [0, 1] range.
Out of range values are clipped. 

2. Histogram Equalization Method:
- First compute the histogram of the image(h(i)) and calculate the cumulative distribution function(f(i)) and applied histogram equalization based on the L in the specified window(H1, W1) to (H2, W2).
Then transfer the RGB color space to luv and map the L that obtained from histogram equalization to the according pixel.
Finally transfer back to RGB color space.

- During the transformation of LUV to XYZ, I add the condition to L == 0 since the dominator of the equation cannot be 0. Apply l'hospital law we have u = (13*uw)/13 and v = (13*vw)/13 if L == 0.

- Since we apply linear scaling to L, the RGB value transfer from XYZ might out of the [0, 1] range.
Out of range values are clipped. 



## Running the Program

Foramt of the execution: 
- linear scaling: python linearscaling.py w1 h1 w2 h2 {Input Image} {Output Image}
- histogram equalization: python histequal.py w1 h1 w2 h2 {Input Image} {Output Image}
Note: The window is specified in terms of the normalized coordinates w1 h1 w2 h2, where the window
upper left point is (w1,h1), and its lower right point is (w2,h2).



## Authors

* **Samuel Chen** - *Initial work*



