from ultralytics import YOLO
import cv2

# import util
from sort.sort import *
from util import get_car#, write_csv
from char_ocr import read_license_plate
# from util import read_license_plate


results = {}

mot_tracker = Sort()

# load models
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')

# load video
# cap = cv2.VideoCapture('WhatsApp Video 2023-11-22 at 10.38.18 AM.mp4')
# cap = cv2.VideoCapture('test_image.jpg')
# cap = cv2.VideoCapture('WhatsApp Video 2023-11-22 at 10.38.19 AM.mp4')
# cap = cv2.VideoCapture('demo_video.mp4')
# cap = cv2.VideoCapture('demo_i_3.mp4')
cap = cv2.VideoCapture('demo_i_4.mp4')

vehicles = [2, 3, 5, 7]

# read frames
frame_nmr = -1
ret = True
cnt = 0
while ret:
    if cv2.waitKey(25) & 0xFF == ord('q'): 
        break
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        try:
            cv2.imshow('Frame', frame)
            results[frame_nmr] = {}
            # detect vehicles
            cv2.imshow('Frame', frame)
            detections = coco_model(frame,verbose=False)[0]
            print("inside model preditction")
            detections_ = []
            for detection in detections.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                print(class_id)
                if int(class_id) in vehicles:
                    detections_.append([x1, y1, x2, y2, score])

            # # track vehicles
            track_ids = mot_tracker.update(np.asarray(detections_))

            # detect license plates
            license_plates = license_plate_detector(frame,verbose=True)[0]
            for license_plate in license_plates.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = license_plate

                # assign license plate to car
                xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
                print(xcar1, ycar1, xcar2, ycar2, car_id)
                if car_id != -1:

                    # crop license plate
                    # license_plate_crop = frame[int(y1)-2:int(y2)+2, int(x1)-2: int(x2)+2, :]
                    license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)
                    cv2.imshow('Frame', frame)

                    # process license plate
                    # license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                    # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                    # read license plate number
                    # license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                    cv2.imwrite('corresponding_frame_image.jpg',frame)
                    cv2.imwrite('corresponding_act_image.jpg',frame[int(y1)-2:int(y2)+2, int(x1)-2: int(x2)+2, :])
                    license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop)

                    if license_plate_text is not None:
                        cv2.putText(frame, license_plate_text, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        cv2.imshow('Frame', frame)
                        results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                      'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                        'text': license_plate_text,
                                                                        'bbox_score': score,
                                                                        'text_score': license_plate_text_score
                                                                        }
                                                    }
                        print(f"{results[frame_nmr][car_id]['license_plate']=}")
                        # raise RuntimeError("Got the results")
        except  RuntimeError as err:
            print(f"finished, {str(err)}")
            break
            cnt+=1
            if cnt>10:
                break
        except Exception as err:
            print(f"Exception as {str(err)}")
            pass

cap.release() 
  
# Closes all the frames 
cv2.destroyAllWindows() 
# write results
# write_csv(results, './test.csv')