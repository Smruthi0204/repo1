if __name__ == '__main__':
    from ultralytics import YOLO

    
    path = "D:\\project1\\object_detection\\pedestrian_auto_dataset\\data.yaml"

    
    model = YOLO("D:\\project1\\object_detection\\yolo11n\\runs\\detect\\train2\\weights\\last.pt")

    
    results = model.train(
        data=path,
        epochs=30,        
        imgsz=320,
        batch=10,
    )
