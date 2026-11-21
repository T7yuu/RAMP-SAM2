"""Native SAM2 automatic-mask evaluation with a low-memory point grid."""
import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import table23_runner_v2 as r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--datasets',nargs='+',default=['ISIC2018','PH2']); ap.add_argument('--output-root',default=str(ROOT/'scripts/V9/table23_experiments')); ap.add_argument('--results',default=str(ROOT/'scripts/V9/table23_experiments/native_sam2_results.json')); ap.add_argument('--device',default='cuda'); ap.add_argument('--points-per-side',type=int,default=8); ap.add_argument('--sam2-repo',default=r'E:/PythonD/ERA/sam2_code'); ap.add_argument('--sam2-config',default=r'E:/PythonD/ERA/sam2_weights/config.yaml'); ap.add_argument('--sam2-checkpoint',default=r'E:/PythonD/ERA/sam2_weights/sam2.1_hiera_base_plus.pt'); a=ap.parse_args(); import torch
    sys.path.insert(0,a.sam2_repo); from sam2.build_sam import build_sam2; from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    device=a.device; model=build_sam2(a.sam2_config,a.sam2_checkpoint,device=device); gen=SAM2AutomaticMaskGenerator(model=model,points_per_side=a.points_per_side,pred_iou_thresh=.8,stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=100)
    sums=[]; per=[]
    for d in a.datasets:
        ps=r.cases(d); out=Path(a.output_root)/d/'Native_SAM2_lowmem'; out.mkdir(parents=True,exist_ok=True)
        for name,p,_ in ps:
            op=out/f'{name}.png'
            if not op.exists():
                arr=np.asarray(Image.open(p).convert('RGB'))
                with torch.inference_mode(): generated=gen.generate(arr)
                area=arr.shape[0]*arr.shape[1]; valid=[x for x in generated if x['area']<.9*area]; chosen=max(valid or generated,key=lambda x:x['area'],default=None); mm=chosen['segmentation'] if chosen else np.zeros(arr.shape[:2],bool); Image.fromarray(mm.astype(np.uint8)*255).save(op)
        s,rows=r.evaluate(d,out,ps); s.update({'dataset':d,'model':'Native SAM2','status':'ok','mode':'inference','points_per_side':a.points_per_side,'device':device}); sums.append(s); per += rows; print(d,s,flush=True)
    Path(a.results).write_text(__import__('json').dumps({'summaries':sums,'per_image':per},ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
