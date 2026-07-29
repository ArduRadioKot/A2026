
import cv2
import numpy as np
import time

WEIGHTS=np.load("digits64_weights.npz")

def relu(x):
    return np.maximum(x,0)

def conv(x,w,b):
    c,h,wid=x.shape
    out=np.zeros((w.shape[0],h,wid),np.float32)

    for oc in range(w.shape[0]):
        for ic in range(c):
            out[oc]+=cv2.filter2D(
                x[ic],
                -1,
                w[oc,ic],
                borderType=cv2.BORDER_CONSTANT
            )
        out[oc]+=b[oc]

    return out


def pool(x):
    c,h,w=x.shape
    out=np.zeros((c,h//2,w//2),np.float32)

    for ch in range(c):
        for y in range(0,h,2):
            for x0 in range(0,w,2):
                out[ch,y//2,x0//2]=np.max(
                    x[ch,y:y+2,x0:x0+2]
                )
    return out


def linear(x,w,b):
    return w@x+b


def predict(img):

    img=cv2.resize(img,(64,64))
    img=img.astype(np.float32)/255
    img=(img-0.1307)/0.3081

    x=img[None]

    x=pool(relu(conv(
        x,
        WEIGHTS['features.0.weight'],
        WEIGHTS['features.0.bias']
    )))

    x=pool(relu(conv(
        x,
        WEIGHTS['features.3.weight'],
        WEIGHTS['features.3.bias']
    )))

    x=pool(relu(conv(
        x,
        WEIGHTS['features.6.weight'],
        WEIGHTS['features.6.bias']
    )))

    x=x.flatten()

    x=relu(linear(
        x,
        WEIGHTS['classifier.1.weight'],
        WEIGHTS['classifier.1.bias']
    ))

    x=linear(
        x,
        WEIGHTS['classifier.4.weight'],
        WEIGHTS['classifier.4.bias']
    )

    return np.argmax(x)


cap=cv2.VideoCapture(0)

while True:

    ok,frame=cap.read()
    if not ok:
        continue

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    digit=cv2.resize(gray,(64,64))

    t=time.time()

    cls=predict(digit)

    fps=1/(time.time()-t)

    cv2.putText(
        frame,
        f'{cls} {fps:.1f} FPS',
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow(
        'camera',
        frame
    )

    if cv2.waitKey(1)==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
