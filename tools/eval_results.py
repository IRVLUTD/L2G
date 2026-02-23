
## evaluate the results using COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



# test RoboTools all
# cocoGt = COCO("../data/Ground_Truth/RoboTools/scene_gt_coco_all.json")
# cocoDt = cocoGt.loadRes("../Output/RoboTools/merged_coco.json")


#test High_Resolution all
cocoGt = COCO("../data/Ground_Truth/High_Resolution/scene_gt_coco_all.json")
cocoDt = cocoGt.loadRes("../data/Results_COCO/High_Resolution/adapter=True_SAM*=True.json")

#test High_Resolution
cocoGt = COCO("../data/Ground_Truth/High_Resolution/scene_gt_coco_hard.json")
cocoDt = cocoGt.loadRes("../data/Results_COCO/High_Resolution/hard.json")

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# get IoU 0.95
# cocoEval.params.iouThrs = [0.95]
# Run the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print(cocoEval.stats)
