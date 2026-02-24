
## evaluate the results using COCO API
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval



# test RoboTools
cocoGt = COCO("../eval_results/Ground_Truth/RoboTools/scene_gt_coco_all.json")
cocoDt = cocoGt.loadRes("../eval_results/Results_COCO/RoboTools/L2G_all.json")

# #test High_Resolution all
# cocoGt = COCO("../eval_results/Ground_Truth/High_Resolution/scene_gt_coco_all.json")
# cocoDt = cocoGt.loadRes("../eval_results/Results_COCO/High_Resolution/L2G_all.json")

# #test High_Resolution hard
# cocoGt = COCO("../eval_results/Ground_Truth/High_Resolution/scene_gt_coco_hard.json")
# cocoDt = cocoGt.loadRes("../eval_results/Results_COCO/High_Resolution/L2G_hard.json")

# #test High_Resolution easy
# cocoGt = COCO("../eval_results/Ground_Truth/High_Resolution/scene_gt_coco_easy.json")
# cocoDt = cocoGt.loadRes("../eval_results/Results_COCO/High_Resolution/L2G_easy.json")

cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

# get IoU 0.95
# cocoEval.params.iouThrs = [0.95]
# Run the evaluation
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

print(cocoEval.stats)
