import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

from detectron2.utils.logger import setup_logger
setup_logger()
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import layoutparser as lp
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import json

#register dataset
from detectron2.data.datasets import register_coco_instances
register_coco_instances("receipts", {}, "../receipttrain/content/datasets/receipts-1.json", "../receipttrain/content/datasets/receipts")

#set config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("receipts", )
#cfg.DATASETS.TEST = ("receipts", )
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4


cfg.MODEL.WEIGHTS = "../receipttrain/output/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.80 # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

#predict based on example image
from detectron2.utils.visualizer import ColorMode

im = cv2.imread("../receipttrain/content/datasets/receipts/bon1.jpg")
outputs = predictor(im)
instance_pred = outputs['instances'].to("cpu")
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.0)
v = v.draw_instance_predictions(instance_pred)
cv2.imshow('',v.get_image()[:, :, ::-1])
cv2.waitKey()





#collect outputs and label names
instance_pred = outputs['instances'].to("cpu")
receiptlabels = {0: "header", 1: "rows", 2: "footer", 3:"totaal"}

#use layout parser to parse data and prep for OCR
layout = lp.Layout()
scores = instance_pred.scores.tolist()
boxes = instance_pred.pred_boxes.tensor.tolist()
labels = instance_pred.pred_classes.tolist()

for score, box, label in zip(scores, boxes, labels):
    x_1, y_1, x_2, y_2 = box

    if receiptlabels is not None:
        label = receiptlabels.get(label, label)

    cur_block = lp.TextBlock(
        lp.Rectangle(x_1, y_1, x_2, y_2),
        type=label,
        score=score)
    layout.append(cur_block)

text_blocks = lp.Layout([b for b in layout if b.type=='header'])
text_blocks = text_blocks + lp.Layout([b for b in layout if b.type=='rows'])
text_blocks = text_blocks + lp.Layout([b for b in layout if b.type=='totaal'])
text_blocks = text_blocks + lp.Layout([b for b in layout if b.type=='footer'])
figure_blocks = lp.Layout([b for b in layout if b.type=='Figure'])

text_blocks = lp.Layout([b for b in text_blocks \
                   if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

h, w = im.shape[:2]

left_interval = lp.Interval(0, w/2*1.05, axis='x').put_on_canvas(im)

left_blocks = text_blocks.filter_by(left_interval, center=True)
left_blocks.sort(key = lambda b:b.coordinates[1])

right_blocks = [b for b in text_blocks if b not in left_blocks]
right_blocks.sort(key = lambda b:b.coordinates[1])

# And finally combine the two list and add the index
# according to the order
text_blocks = lp.Layout([b.set(id = idx) for idx, b in enumerate(left_blocks + right_blocks)])

lp.draw_box(im, text_blocks,
            box_width=3,
            show_element_id=True).show()


#go for OCR
ocr_agent = lp.TesseractAgent(languages='eng+nld')
    # Initialize the tesseract ocr engine. You might need
    # to install the OCR components in layoutparser:
    # pip install layoutparser[ocr]



receipt_dict = []
for block in text_blocks:
   segment_image = (block
                     .pad(left=3, right=3, top=3, bottom=3)
                     .crop_image(im))
   text = ocr_agent.detect(segment_image)
   block.set(text=text, inplace=True)
   print(block.type, text, end='\n---\n')
   receipt_dict  += [[block.type,text]]

receipt_dict = json.dumps(receipt_dict)

print(receipt_dict)
