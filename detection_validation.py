if __name__ == '__main__':
    from ultralytics import YOLO


    model = YOLO("C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\identification\\runs\\detect\\train4\\weights\\best.pt")  

    model.val(data="C:\\Users\\smrut\\OneDrive\\Desktop\\SEM Y\\Internship\\project1 BC\\identification\\Pedestrian and Auto.v2i.yolov8\\data.yaml")  
   