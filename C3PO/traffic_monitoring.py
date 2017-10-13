from darknet import *
import cv2

if __name__ == "__main__":
    net = load_net(b"cfg/yolo.cfg", b"weights/yolo.weights", 0)
    meta = load_meta(b"cfg/coco.data")
    r = detect(net, meta, b"data/dog.jpg")
    
    print (r[0][0]) # label
    print (r[0][1]) # confidence
    print (r[0][2]) # box coordinates

    print (r[1][0]) # label
    print (r[1][1]) # confidence
    print (r[1][2]) # box coordinates

    r = detect(net, meta, b"data/scream.jpg")
    print (r)

    r = detect(net, meta, b"data/eagle.jpg")
    print (r)


def pipeline(frame):
    return frame

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy