#!/usr/bin/env python3
import argparse, os, random, shutil
from pathlib import Path

p=argparse.ArgumentParser(); p.add_argument('--real-train',type=Path,required=True); p.add_argument('--real-val',type=Path,required=True)
p.add_argument('--synthetic-root',type=Path,required=True); p.add_argument('--output-root',type=Path,required=True)
p.add_argument('--majority-cap',type=int,required=True); p.add_argument('--seed',type=int,default=42); a=p.parse_args(); random.seed(a.seed)
exts={'.jpg','.jpeg','.png','.bmp'}
def imgs(d): return sorted(x for x in d.iterdir() if x.is_file() and x.suffix.lower() in exts)
def link(src,dst): dst.parent.mkdir(parents=True,exist_ok=True); os.symlink(src,dst)
classes=sorted(x.name for x in a.real_train.iterdir() if x.is_dir()); counts={c:len(imgs(a.real_train/c)) for c in classes}
median=sorted(counts.values())[len(counts)//2]; majority={c for c,n in counts.items() if n>median}
if a.output_root.exists(): shutil.rmtree(a.output_root)
for c in classes:
    for x in imgs(a.real_val/c): link(x,a.output_root/'val'/c/x.name)
    for x in imgs(a.real_train/c): link(x,a.output_root/'train'/c/x.name)
    syn=imgs(a.synthetic_root/c/'images')
    if c in majority: syn=random.sample(syn,min(a.majority_cap,len(syn)))
    for i,x in enumerate(syn): link(x,a.output_root/'train'/c/f'syn_{i:06d}{x.suffix.lower()}')
print('real_counts',counts,'majority',sorted(majority),'created',a.output_root)
