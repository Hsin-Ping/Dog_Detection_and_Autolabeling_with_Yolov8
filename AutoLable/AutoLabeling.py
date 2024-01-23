from ultralytics import YOLO
import cv2
import os
import glob


def img_size_normalize(img):
    if not img.shape == (640, 640, 3):
        print(f"original resolution is {img.shape}, resize to 640*640*3")
        img = cv2.resize(img,(640,640))
    return img
 
def GetNewDatasPath(NewDatasetFolder):
    NewDataPaths = list(glob.glob(os.path.join(NewDatasetFolder,"*.jpg")))
    return NewDataPaths

def output_obj_labes(NewDatasetFolder, img_name, objs):
    labelfile = os.path.join(NewDatasetFolder, img_name[:-4]+".txt")
    objs = list(map(lambda x:" ".join(x.split(" ")[:-1]), objs))
    with open(labelfile, "w") as f:
        for obj in objs:
            f.write(obj)
            f.write("\n")
            
def draw_conf(img, pts1, conf):
    p1_x, p1_y = pts1
    cv2.rectangle(img, (p1_x, p1_y-20), (p1_x+40, p1_y), (0,0,255), -1)
    cv2.putText(img, str(conf), (p1_x, p1_y-5), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,255,255), thickness=2)
    
def show_prediction_result(img, objs):
    img_show = img.copy()
    img_h, img_w, _ = img.shape
    for obj in objs:
        print(obj.split(" ")[1:])
        x, y, w, h, conf = list(map(lambda x:float(x),obj.split(" ")[1:]))
        xx, yy, ww, hh = x*img_w, y*img_h, w*img_w, h*img_h
        pts1 = (int(xx-ww//2), int(yy-hh//2))
        pts2 = (int(xx+ww//2), int(yy+hh//2))
        right_bot = (int(xx-ww//2), int(yy+hh//2))
        cv2.rectangle(img_show, pts1, pts2, (0,0,255),3)
        draw_conf(img_show, right_bot, conf)
    cv2.imshow("Prediction_Result", img_show)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def prediction(NewDatasetFolder, model, NewDataPaths, show=False):
    for img_path in NewDataPaths:
        img_name = img_path.split('/')[-1]
        img = cv2.imread(img_path)
        img = img_size_normalize(img)
        img_h, img_w, _ = img.shape
        results = model.predict(img,iou=0.5,conf=0.4)
        objs = []
        # Extract bounding boxes(after normalization), classes
        for box in results[0].boxes:
            x, y, w, h = box.xywhn.tolist()[0]
            classes = int(box.cls.tolist()[0])
            conf = round(box.conf.tolist()[0],2)
            obj = " ".join([str(classes), str(x), str(y), str(w), str(h), str(conf)])
            objs.append(obj)
        output_obj_labes(NewDatasetFolder, img_name, objs)
        if show:
            show_prediction_result(img, objs)

if __name__ == "__main__":

    modelpath = "../model/best.pt"
    model = YOLO(modelpath)

    NewDatasetFolder = "MyDataset"
    
    NewDataPaths = GetNewDatasPath(NewDatasetFolder)
    
    prediction(NewDatasetFolder, model, NewDataPaths, show=True)
    
    
