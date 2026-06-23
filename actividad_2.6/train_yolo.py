from ultralytics import YOLO                                                                          
model = YOLO('yolov8s.pt')
model.train(                                                                                          
    data='signals/merged_dataset/data.yaml',                  
    epochs=100,
    imgsz=640,
    batch=8,                                                                                          
    augment=True,
    device=0,                                                                                         
    project='signals',                                        
    name='train_merged',
    exist_ok=True,
)
