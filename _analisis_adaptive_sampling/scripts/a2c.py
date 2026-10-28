import pandas as pd, numpy as np, os, time, sys
t0=time.time(); R=os.path.expanduser('~/mnt/SKRIPSI')
o=pd.read_csv(f'{R}/stage_8_2_difficulty_clean_20261020/train_object_difficulty.csv')
m=pd.read_csv(f'{R}/stage_8_2_difficulty_clean_20261020/train_image_difficulty_manifest.csv')
P=pd.read_pickle('/tmp/ana/P.pkl'); P['image_id']=P.image_id.astype(str)
idx=P.groupby('image_id').indices
B=P[['x1','y1','x2','y2']].values; CF=P.confidence.values; CL=P.class_id.values
dims=dict(zip(m.image_id,zip(m.img_width,m.img_height)))
LD=f'{R}/cassava-v7-standardized/train/labels'
S,K=int(sys.argv[1]),int(sys.argv[2])
imgs=sorted(o.image_id.unique())[S::K]
res={}
for img in imgs:
    W,H=dims[img]
    L=np.array([list(map(float,l.split()[:5])) for l in open(f'{LD}/{os.path.splitext(img)[0]}.txt').read().split('\n') if l.strip()])
    g=np.stack([(L[:,1]-L[:,3]/2)*W,(L[:,2]-L[:,4]/2)*H,(L[:,1]+L[:,3]/2)*W,(L[:,2]+L[:,4]/2)*H],1); gc=L[:,0].astype(int)
    if img not in idx:
        for k in range(len(L)): res[(img,k)]=(0,0,-1,0,0)
        continue
    j=idx[img]; b=B[j]; cf=CF[j]; cl=CL[j]
    x1=np.maximum(g[:,None,0],b[None,:,0]);y1=np.maximum(g[:,None,1],b[None,:,1]);x2=np.minimum(g[:,None,2],b[None,:,2]);y2=np.minimum(g[:,None,3],b[None,:,3])
    inter=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None); ag=((g[:,2]-g[:,0])*(g[:,3]-g[:,1]))[:,None]; ab=((b[:,2]-b[:,0])*(b[:,3]-b[:,1]))[None,:]
    I=inter/(ag+ab-inter+1e-9)
    for k in range(len(L)):
        same=(cl==gc[k])&(I[k]>=0.5); anyc=I[k]>=0.5
        bs=float(cf[same].max()) if same.any() else 0.0
        if anyc.any():
            jj=np.argmax(np.where(anyc,cf,-1)); ba=float(cf[jj]); bac=int(cl[jj])
        else: ba=0.0; bac=-1
        res[(img,k)]=(bs,ba,bac,int((I[k]>=0.1).sum()),float(I[k].max()))
X=pd.DataFrame([(a,b_,*v) for (a,b_),v in res.items()],columns=['image_id','gt_id','best_same_conf50','best_any_conf50','best_any_cls','n_cand_iou10','max_iou'])
X.to_pickle(f'/tmp/ana/shard_{S}.pkl')
print('done shard',S,len(X),round(time.time()-t0,1),'s',flush=True)
