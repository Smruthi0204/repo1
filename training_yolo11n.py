if __name__ == '__main__':
    from ultralytics import YOLO
    import os

    path = "D:\\project1\\object_detection\\pedestrian_auto_dataset\\data.yaml"
   

    model = YOLO('yolo11n.pt')

    results = model.train(data=path,
                          epochs=30,
                          imgsz=320, batch=4)
