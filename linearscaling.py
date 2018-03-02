import cv2
import numpy as np
import sys

def invgamma(v):
    if(v < 0.03928):
        return v/12.92
    else:
        return np.power((v+0.055)/1.055, 2.4)

def gamma(d):
    if d < 0.00304:
        return 12.92*d
    else:
        return (1.055*np.power(d, 1/2.4))-0.055
def limit(n):
    if n >= 1:
        return 1
    elif n <= 0:
        return 0
    else:
        return n
#Conversion Matrix

matrix = [[0.412453, 0.357580, 0.180423],
          [0.212671, 0.715160, 0.072169],
          [0.019334, 0.119193, 0.950227]]

matrix2 = [[3.240479, -1.53715, -0.498535],
           [-0.969256, 1.875991, 0.041556],
           [0.055648, -0.204043, 1.057311]]

if(len(sys.argv) != 7) :
    print(sys.argv[0], ": takes 6 arguments. Not ", len(sys.argv)-1)
    print("Expecting arguments: w1 h1 w2 h2 ImageIn ImageOut.")
    print("Example:", sys.argv[0], " 0.2 0.1 0.8 0.5 fruits.jpg out.png")
    sys.exit()

w1 = float(sys.argv[1])
h1 = float(sys.argv[2])
w2 = float(sys.argv[3])
h2 = float(sys.argv[4])
name_input = sys.argv[5]
name_output = sys.argv[6]

if(w1<0 or h1<0 or w2<=w1 or h2<=h1 or w2>1 or h2>1) :
    print(" arguments must satisfy 0 <= w1 < w2 <= 1, 0 <= h1 < h2 <= 1")
    sys.exit()

inputImage = cv2.imread(name_input, cv2.IMREAD_COLOR)
if(inputImage is None) :
    print(sys.argv[0], ": Failed to read image from: ", name_input)
    sys.exit()

rows, cols, bands = inputImage.shape # bands == 3
W1 = round(w1*(cols-1))
H1 = round(h1*(rows-1))
W2 = round(w2*(cols-1))
H2 = round(h2*(rows-1))

# The transformation should be based on the
# historgram of the pixels in the W1,W2,H1,H2 range.
# The following code goes over these pixels

outputImage = np.copy(inputImage)

#Transform rgb to luv
minL = 100
maxL = 0
#Calculate max, min
for i in range(H1, H2) :
    for j in range(W1, W2) :
        b, g, r = inputImage[i, j]
        b = b / 255
        g = g / 255
        r = r / 255
        b = invgamma(b)
        g = invgamma(g)
        r = invgamma(r)
        xyzmatrix = np.dot(matrix, [r, g, b])
        x = (xyzmatrix[0])
        y = (xyzmatrix[1])
        z = (xyzmatrix[2])
        uw = (4 * 0.95) / (0.95 + 15 * 1.0 + 3 * 1.09)
        vw = (9 * 1.0) / (0.95 + 15 * 1.0 + 3 * 1.09)
        t = y / 1.0

        L = float(116 * np.power(t, 1 / 3.0) - 16.0 if t > 0.008856 else 903.3 * t)
        d = float(x + 15.0 * y + 3.0 * z)
        if d != 0:
            ui = 4.0 * x / d
            vi = 9.0 * y / d
            u = 13 * L * (ui - uw)
            v = 13 * L * (vi - vw)
        else:
            ui = 4
            vi = 9
            u = 13 * L * (ui - uw)
            v = 13 * L * (vi - vw)
        if(L < minL):
            minL = L
        if(L > maxL):
            maxL = L

for i in range(rows):
    for j in range(cols):
        b, g, r = inputImage[i, j]
        b = b/255
        g = g/255
        r = r/255
        b = invgamma(b)
        g = invgamma(g)
        r = invgamma(r)

        xyzmatrix = np.dot(matrix, [r, g, b])
        x = (xyzmatrix[0])
        y = (xyzmatrix[1])
        z = (xyzmatrix[2])

        #xyz to luv
        uw = (4*0.95)/(0.95+15*1.0+3*1.09)
        vw = (9*1.0)/(0.95+15*1.0+3*1.09)
        t = y/1.0

        if t > 0.008856:
            L = float(116 * np.power(t, 1 / 3.0) - 16.0)
        else:
            L = 903.3 * t

        d = float(x + 15.0*y + 3.0*z)

        if d != 0:
            u_dash = 4.0*x/d
            v_dash = 9.0*y/d
            u = 13*L*(u_dash - uw)
            v = 13*L*(v_dash - vw)
        else:
            u_dash = 4
            v_dash = 9
            u = 13*L*(u_dash - uw)
            v = 13*L*(v_dash - vw)

        #Linear Scaling
        if (L > maxL):
            L = 100
        elif (L < minL):
            L = 0
        else:
            L = (int)(((L - minL)*100.0)/(maxL - minL))

        #luv to xyz
        uw = (4*0.95)/(0.95+15*1.0+3*1.09)
        vw = (9*1.0)/(0.95+15*1.0+3*1.09)

        if( L != 0):
            u_dash = (u+13*uw*L)/(13*L)
            v_dash = (v+13*vw*L)/(13*L)
        else:
            u_dash = (13*uw)/13
            v_dash = (13*vw)/13

        if L > 7.9996:
            y = np.power((L+16)/116, 3)*1.0
        else:
            y = L/903.3

        if(v_dash != 0):
            x = y*2.25*(u_dash/v_dash)
            z = (y*(3.0 - (0.75*u_dash) - (5.0*v_dash))) / v_dash
        else:
            x = 0
            z = 0

        #xyz to rgb
        rgbmatrix = np.dot(matrix2, [x, y, z])

        R = limit(rgbmatrix[0])
        G = limit(rgbmatrix[1])
        B = limit(rgbmatrix[2])

        R = gamma(R)*255
        G = gamma(G)*255
        B = gamma(B)*255

        outputImage[i, j] = [int(B), int(G), int(R)]

cv2.imshow("Input:", inputImage)

cv2.imshow("output:", outputImage)
cv2.imwrite(name_output, outputImage);

# wait for key to exit
cv2.waitKey(0)
cv2.destroyAllWindows()
