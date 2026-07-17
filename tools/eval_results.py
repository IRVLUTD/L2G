
## evaluate the results using COCO API
import argparse
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

GT_ROOT = Path("../eval_results/Ground_Truth")
DT_ROOT = Path("../eval_results/Results_COCO")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate L2G detection results with the COCO API.")
    parser.add_argument(
        "--dataset",
        choices=["High_Res", "RoboTools"],
        default="High_Res",
        help="Which dataset's results to evaluate.",
    )
    parser.add_argument(
        "--split",
        choices=["hard", "easy", "all"],
        default="all",
        help="High_Res split to evaluate. Ignored for RoboTools, which only has 'all'.",
    )
    args = parser.parse_args()

    if args.dataset == "RoboTools":
        gt_path = GT_ROOT / "RoboTools" / "scene_gt_coco_all.json"
        dt_path = DT_ROOT / "RoboTools" / "L2G_all.json"
    else:
        gt_path = GT_ROOT / "High_Resolution" / f"scene_gt_coco_{args.split}.json"
        dt_path = DT_ROOT / "High_Resolution" / f"L2G_{args.split}.json"

    print(f"Evaluating dataset={args.dataset} split={args.split}")
    print(f"  GT: {gt_path}")
    print(f"  Dt: {dt_path}")

    cocoGt = COCO(str(gt_path))
    cocoDt = cocoGt.loadRes(str(dt_path))

    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")

    # get IoU 0.95
    # cocoEval.params.iouThrs = [0.95]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print(cocoEval.stats)


if __name__ == "__main__":
    main()
