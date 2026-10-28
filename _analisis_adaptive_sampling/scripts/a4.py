import pandas as pd, numpy as np, json
o=pd.read_pickle('obj_validity.pkl')
import os; R=os.path.expanduser('~/mnt/SKRIPSI')
m=pd.read_csv(f'{R}/stage_8_2_difficulty_clean_20261020/train_image_difficulty_manifest.csv')
rng=np.random.RandomState(7)
hard=m[(m.d_image>=0.8)&(m.has_gt==1)].image_id.values; easy=m[m.d_image<=m.d_image.quantile(.1)].image_id.values
# avoid picking twins: use unique base prefix
def pick(arr,k):
    seen=set(); out=[]
    for f in rng.permutation(arr):
        b=f.split('_jpg')[0]
        if b in seen: continue
        seen.add(b); out.append(f)
        if len(out)==k: break
    return out
H=pick(hard,10); E=pick(easy,4)
P=pd.read_pickle('P.pkl'); P['image_id']=P.image_id.astype(str)
sel=P[P.image_id.isin(H+E)&(P.confidence>=0.25)]
d={'hard':H,'easy':E,'dimg':{f:float(m.set_index('image_id').d_image[f]) for f in H+E},'preds':sel.round(2).values.tolist()}
json.dump(d,open('vis_sel.json','w')); print(json.dumps({'hard':H,'easy':E}))
print('npred',len(sel))
