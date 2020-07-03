import time
import gluoncv as gcv
from gluoncv.utils import try_import_cv2
cv2 = try_import_cv2()
import mxnet as mx

net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)

cap = cv2.VideoCapture(0)
time.sleep(1)

xes = None
NUM_FRAMES = 200
for i in range(NUM_FRAMES):
    ret, frame = cap.read()

    frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
    rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

    class_IDs, scores, bounding_boxes = net(rgb_nd)

    img = gcv.utils.viz.cv_plot_bbox(frame, bounding_boxes[0], scores[0], class_IDs[0], class_names=net.classes)
    gcv.utils.viz.cv_plot_image(img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()