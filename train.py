# -*- coding: utf-8 -*-

from ultralytics import YOLO
from ultralytics import models

model = YOLO("yolo11m_CF.yaml")
#model = YOLO('runs/detect/train3/weights/last.pt')

#model = YOLO('pre.pt')
results = model.train(resume=True, data="datasets/AITOD.yaml", workers=16, epochs=650, batch=16, amp=True)

results = model.val()

#results = model("E:\yolo\1.jpg")
