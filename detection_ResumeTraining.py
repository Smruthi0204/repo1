if __name__ == '__main__':
    from ultralytics import YOLO

    
    path = "C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\identification\\Pedestrian and Auto.v2i.yolov8\\data.yaml"

    
    model = YOLO("C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\identification\\runs\\detect\\train4\\weights\\last.pt")

    
    results = model.train(
        data=path,
        epochs=10,        
        imgsz=320,
        batch=10,
        resume=True
    )
