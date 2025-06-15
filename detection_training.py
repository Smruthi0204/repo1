if __name__ == '__main__':
    from ultralytics import YOLO
    import os

    path = "C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\identification\\Pedestrian and Auto.v2i.yolov8\\data.yaml"
   

    model = YOLO('yolov8n.pt')

    results = model.train(data=path,
                          epochs=4,
                          imgsz=320, batch=4)
