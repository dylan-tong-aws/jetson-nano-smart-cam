import time
import imutils
import os

import cv2
import numpy as np
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon.data.vision import transforms
from dlr import DLRModel

from cv_obj_tracker import CVObjectTracker

USE_GPU = True
USE_DLR = False

if USE_GPU :
    try:
        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
        CTX = [mx.gpu(0)]
        print('GPU device is available')
        USE_GPU = True
        os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
    except:
        CTX = [mx.cpu()]
        print("Unable to locate GPU. Using CPU.")
else:
    CTX = [mx.cpu()]     

#if USE_DLR :
#OD_MODEL_DIR = "./models/yolov3/qr-bottle/dlr/180"
#SRGAN_MODEL_DIR = "./models/srgan/dlr"
#else :
OD_MODEL_DIR = "./models/yolov3/qr-bottle/symbolic"
SRGAN_MODEL_DIR = "./models/srgan/symbolic"

class ResnetBlock(gluon.nn.HybridBlock):
    def __init__(self):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.HybridSequential()
        with self.name_scope():
            self.conv_block.add(
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm()
            )

    def hybrid_forward(self, F, x,*args, **kwargs):
        out = self.conv_block(x)
        return out + x

class SubpixelBlock(gluon.nn.HybridBlock):
    def __init__(self, shape):
        super(SubpixelBlock, self).__init__()
        self.conv = nn.Conv2D(256, kernel_size=3, strides=1,padding=1)
        self.relu = nn.Activation('relu')
        self.shape = shape

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.conv(x)
        x = x.transpose([0, 2, 3, 1])
        batchsize,height,width,depth = self.shape
        x = x.reshape((batchsize, height , width, 2, 2, int(depth / 4)))
        x = x.transpose([0, 1,3,2,4,5])
        x = x.reshape((batchsize, height * 2, width * 2, int(depth / 4)))
        x = x.transpose([0, 3, 1, 2])
        x = self.relu(x)
        return x

class SRGenerator(gluon.nn.HybridBlock):
    def __init__(self):
        super(SRGenerator, self).__init__()
        self.conv1 = nn.Conv2D(64, kernel_size=3, strides=1,padding=1,activation='relu')
        self.res_block = nn.HybridSequential()
        with self.name_scope():
            for i in range(16):
                self.res_block.add(
                    ResnetBlock()
                )

            self.res_block.add(
                nn.Conv2D(64, kernel_size=3, strides=1,padding=1,use_bias=False),
                nn.BatchNorm()
            )
        self.subpix_block1 = SubpixelBlock((1, 75, 75, 256))
        self.subpix_block2 = SubpixelBlock((1, 150, 150, 256))
        
        self.conv4 = nn.Conv2D(3,kernel_size=1,strides=1,activation='tanh')

    def hybrid_forward(self, F, x,*args, **kwargs):
        x = self.conv1(x)
        out = self.res_block(x)
        x = out + x
        x = self.subpix_block1(x)
        x = self.subpix_block2(x)
        x = self.conv4(x)
        return x

def load_sym_model(sym_f, param_f, model_dir=OD_MODEL_DIR, use_gpu=USE_GPU) :

    if use_gpu :
        device = mx.gpu()
    else :
        device = mx.cpu()

    sym_file = os.path.join(model_dir, sym_f)
    param_file = os.path.join(model_dir, param_f)
    sym_model = gluon.nn.SymbolBlock.imports(sym_file, ['data'], param_file, ctx=device)

    return sym_model

def load_neo_model(model_dir=OD_MODEL_DIR, use_gpu=USE_GPU) :

    if use_gpu :
        device = "gpu"
    else :
        device = "cpu"

    dlr_model = DLRModel(model_dir, device)
    return dlr_model

#if USE_DLR :
#OD_MODEL = load_neo_model(OD_MODEL_DIR)
#SR_MODEL = load_neo_model(SRGAN_MODEL_DIR)
SR_MODEL = load_sym_model("netG_epoch_20000-symbol.json","netG_epoch_20000-0000.params", SRGAN_MODEL_DIR, True)
#else :
OD_MODEL = load_sym_model("mobilenet1.0_custom-symbol.json","mobilenet1.0_custom-0000.params",OD_MODEL_DIR, True)
#    SR_MODEL = load_sym_model(SRGAN_MODEL_DIR, False)

INFERENCE_START = 0.0
INFERENCE_END = 0.0
CAPTURE_WIDTH = 3200
CAPTURE_HEIGHT = 1800
#CAPTURE_WIDTH = 1280
#CAPTURE_HEIGHT = 720
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
FPS = 23
#FPS = 60
THRESHOLD = 0.7
MAX_OBJECTS = 3
MIN_DIST = ((CAPTURE_WIDTH+CAPTURE_HEIGHT)/2) * 0.2
SCALE_FACTOR = 10
ID_COUNTER = 0
CURRENT_SELECTION = ID_COUNTER
TRACK_N_FRAMES = 20
FOCUS_FN_WEIGHT = 8.134311
FOCUS_FN_BIAS = 258.87924
QR_CODE_WIDTH_CM = 1.8
QR_CODE_HEIGHT_CM = 1.8
FOCUS_FN_FEATURE_SCALE = 10000
ZOOM_BOX_SIZE = 75
ZOOM_ZONE = ZOOM_BOX_SIZE * 1.5
WINDOW_NAME = 'QR Detect'
SNAPSHOT_ID = 120   
QR_ID = [1,4]
DATA_DIR = "./snapshots-chk-nuggets"

def focusing(val):
	value = (val << 4) & 0x3ff0
	data1 = (value >> 8) & 0x3f
	data2 = value & 0xf0
	os.system("i2cset -y 6 0x0c %d %d" % (data1,data2))

def laplacian(img):
	img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
	img_sobel = cv2.Laplacian(img_gray,cv2.CV_16U)
	return cv2.mean(img_sobel)[0]

def local_endpoint(x, net=OD_MODEL) :

    global INFERENCE_START
    global INFERENCE_END

    if USE_DLR :
        x = {"data": x.asnumpy()}

        INFERENCE_START = time.time()
        cid, score, bbox = net.run(x)
        INFERENCE_END = time.time()
        m = np.concatenate((cid,score,bbox),axis=2)
        predictions = m[0]
    else :

        if USE_GPU :
            x = x.copyto(mx.gpu())

        INFERENCE_START = time.time()
        cid, score, bbox = net(x)
        INFERENCE_END = time.time()
        m = nd.concat(cid,score,bbox,dim=2)
        predictions = m[0].asnumpy()
    
    return predictions

def imresize(src, w, h, interp=1):
    
    from mxnet.image.image import _get_interp_method as get_interp
    oh, ow, _ = src.shape
    return mx.image.imresize(src, w, h, interp=get_interp(interp, (oh, ow, h, w)))

def resize_short_within(src, short, max_size, mult_base=1, interp=2):

    from mxnet.image.image import _get_interp_method as get_interp
    h, w, _ = src.shape
    im_size_min, im_size_max = (h, w) if w > h else (w, h)
    scale = float(short) / float(im_size_min)
    if np.round(scale * im_size_max / mult_base) * mult_base > max_size:
        # fit in max_size
        scale = float(np.floor(max_size / mult_base) * mult_base) / float(im_size_max)
    new_w, new_h = (int(np.round(w * scale / mult_base) * mult_base),
                    int(np.round(h * scale / mult_base) * mult_base))
    return imresize(src, new_w, new_h, interp=get_interp(interp, (h, w, new_h, new_w)))

def transform_test(imgs, short=416, max_size=1024, stride=1, mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
  
    if isinstance(imgs, mx.nd.NDArray):
        imgs = [imgs]
    for im in imgs:
        assert isinstance(im, mx.nd.NDArray), "Expect NDArray, got {}".format(type(im))

    tensors = []
    origs = []
    for img in imgs:
        img = resize_short_within(img, short, max_size, mult_base=stride)
        orig_img = img.asnumpy().astype('uint8')
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        tensors.append(img.expand_dims(0))
        origs.append(orig_img)
    if len(tensors) == 1:
        return tensors[0], origs[0]
    return tensors, origs

#3280x2464 21 fps
#1920x1080 30 fps
#1280x720 60 fps
def gstreamer_pipeline (capture_width=CAPTURE_WIDTH, capture_height=CAPTURE_HEIGHT, display_width=DISPLAY_WIDTH, display_height=DISPLAY_HEIGHT, framerate=FPS, flip_method=0) :
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method))

def focus_on_selection(h, w) :

    X = ((h/QR_CODE_HEIGHT_CM)*(w/QR_CODE_WIDTH_CM))/FOCUS_FN_FEATURE_SCALE
    focal_distance = int((X*FOCUS_FN_WEIGHT) + FOCUS_FN_BIAS)
    focusing(focal_distance)

    return focal_distance   

def sr_zoom_snapshot(img, xmin, ymin, xmax, ymax, transform, srmodel=SR_MODEL, device=mx.gpu()) :
    
    w = xmax - xmin
    h = ymax - ymin
    side = max(w,h)

    zm_area = img[ymin:ymin+side,xmin:xmin+side]
    lr_img = imutils.resize(zm_area, width=ZOOM_BOX_SIZE*4)
    zm_area = imutils.resize(zm_area, width=ZOOM_BOX_SIZE)
    zm_area = mx.nd.array(zm_area)
    zm_area = transform(zm_area)
    zm_area = zm_area.expand_dims(0).as_in_context(device)
    #zm_area = np.expand_dims(zm_area, axis=0)
    
    start= time.time()
    sr_img = srmodel(zm_area) 
    #sr_img = srmodel.run(zm_area)
    end=time.time()
    #print(sr_img.shape)
    print("SR model inference time {}".format(end-start))
    sr_img = mx.nd.array(sr_img)
    sr_img = mx.nd.squeeze(sr_img)
    sr_img = sr_img.transpose([1,2,0]).asnumpy()
    
    #sr_img = np.squeeze(sr_img, axis=1)
    #sr_img = np.transpose(sr_img, axes=[1,2,0])

    return sr_img, lr_img

def qr_detect() :

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    font = cv2.FONT_HERSHEY_PLAIN
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME,DISPLAY_WIDTH,DISPLAY_HEIGHT)
    cv2.moveWindow(WINDOW_NAME,0,0)
    focal_distance = 10
    objtracker = CVObjectTracker(SCALE_FACTOR, MIN_DIST, THRESHOLD, \
        MAX_OBJECTS,TRACK_N_FRAMES, CVObjectTracker.EUCLIDIAN_LAST_N_FRAMES)

    if cap.isOpened():

        transform_fn = transforms.Compose([ transforms.ToTensor(), \
                                            transforms.Normalize((0.5, 0.5, 0.5), \
                                            (0.5, 0.5, 0.5)),])

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

        while cv2.getWindowProperty(WINDOW_NAME,0) >= 0:

            ret, img = cap.read()
            #resized = imutils.resize(img, width=320)
            #BUFFER, im = transform_test(mx.nd.array(resized), 180)
            #response = local_endpoint(BUFFER)
            #obj_map, obj_selection, active_ids = objtracker.euclidian_get_ids(response)
            
            keyCode = cv2.waitKey(30) & 0xff
            # Stop the program on the ESC key
            if keyCode == 27:
                break
            elif keyCode == 9: # Tab key to toggle selection
                obj_selection = objtracker.get_next_selection(active_ids)
            elif keyCode == 49:
   
                if len(active_ids) > 0 :
                    _,det,_ = active_ids[obj_selection]

                if det :
                    
                    xmin, ymin, xmax, ymax = int(det[2]), int(det[3]), int(det[4]), int(det[5])

                    if xmin > 0 or xmax > 0 or ymin > 0 or ymax > 0 :
                        startzm= time.time()
                        sr_img, lr_img = sr_zoom_snapshot(img,xmin,ymin,xmax,ymax,transform_fn)
                        endzm=time.time()
                        print("Zoom time was {}.".format(endzm - startzm))
                        cv2.namedWindow('Super Res Zoom', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Super Res Zoom',ZOOM_BOX_SIZE*8,ZOOM_BOX_SIZE*8)
                        cv2.imshow('Super Res Zoom',sr_img)

            elif keyCode == 50:
   
                cv2.namedWindow('Low Res Zoom', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('Low Res Zoom',ZOOM_BOX_SIZE*8,ZOOM_BOX_SIZE*8)
                cv2.imshow('Low Res Zoom',lr_img)

            elif keyCode == 51:

                global SNAPSHOT_ID
                #cv2.imwrite(os.path.join(DATA_DIR,"qr-nano-{}-{}.png").format(QR_ID,SNAPSHOT_ID),img)
                cv2.imwrite(os.path.join(DATA_DIR,"chk-nuggets-{}.png").format(SNAPSHOT_ID),img)                
                SNAPSHOT_ID = SNAPSHOT_ID + 1                   
                   
#            for oid in active_ids :

 #               _, det, score = active_ids[oid] 
 #               xmin, ymin, xmax, ymax = int(det[2]), int(det[3]), int(det[4]), int(det[5])

  #              if oid == obj_selection :
                    
   #                 h = ymax-ymin
   #                 w = xmax-xmin

    #                if w <= ZOOM_ZONE or h <= ZOOM_ZONE :
     #                   bb_color = (0, 0, 255)
     #               else :
     #                   bb_color = (0, 255, 0)

     #               focal_distance = focus_on_selection(h,w)
     #           else :
        #              bb_color = (64,64,64)

        #        cv2.rectangle(img,(xmin,ymin),(xmax,ymax),bb_color,5)
        #        cv2.putText(img, "ID: {} Score: {:.3f}".format(oid, score), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, bb_color, 5)
        
      #      cv2.putText(img, 'inference time: {:.5f}'.format(INFERENCE_END-INFERENCE_START), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
            #cv2.putText(img, 'fd: {}'.format(focal_distance), (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)
            #cv2.putText(img, 'val: {:.2f}'.format(val), (50, 150),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 5)

            cv2.imshow(WINDOW_NAME,img)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")

if __name__ == '__main__':
    qr_detect()