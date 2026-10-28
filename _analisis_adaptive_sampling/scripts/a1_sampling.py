import pandas as pd, numpy as np, os
spearman=lambda a,b: pd.Series(a).rank().corr(pd.Series(b).rank())
R=os.path.expanduser('~/mnt/SKRIPSI')
m=pd.read_csv(f'{R}/stage_8_2_difficulty_clean_20261020/train_image_difficulty_manifest.csv')
o=pd.read_csv(f'{R}/stage_8_2_difficulty_clean_20261020/train_object_difficulty.csv')
g=pd.read_csv(f'{R}/cassava-v7-stage-6/resplit_group_manifest.csv',encoding='utf-8-sig')
N=len(m); p=m.p_sampling.values; p=p/p.sum(); w=N*p
print('N',N,'sum p',m.p_sampling.sum())
print('N*p quantiles min/p10/p50/p90/max:',np.round(np.quantile(w,[0,.1,.5,.9,1]),3),'max/min',round(w.max()/w.min(),2))
print('ESS ratio',round(1/(p**2).sum()/N,4))
s=np.sort(p)[::-1]; print('mass top10%%: %.3f  top25%%: %.3f'%(s[:N//10].sum(), s[:N//4].sum()))
hard=m.d_image>=0.8; print('hard imgs',hard.sum(),'mass',round(p[hard].sum(),3),'(uniform would be',round(hard.mean(),3),')')
print('Unique coverage/epoch: weighted-repl %.3f | uniform-repl %.3f | shuffle(EXP-0) 1.000'%(np.mean(1-(1-p)**N), 1-(1-1/N)**N))
E=30
print('Images never drawn in 30 epochs (expected):', round(np.sum((1-p)**(E*N)),1))
print('Expected exposures over 30 ep (EXP-1): p10 %.1f p50 %.1f p90 %.1f max %.1f | EXP-0: 30 each'%tuple(np.quantile(E*w,[.1,.5,.9,1])))
# bottom: easy images exposure
easy=m.d_image<=np.quantile(m.d_image,.25); print('easiest 25%% imgs mean exposures %.1f; hardest 10%% %.1f'%((E*w[easy]).mean(),(E*w[hard]).mean()))
print('has_gt=0 images:',(m.has_gt==0).sum(),'mean D',round(m.d_image[m.has_gt==0].mean(),3))
# object-level weights
wm=dict(zip(m.image_id,w)); o['w']=o.image_id.map(wm)
def comp(col):
    a=o[col].value_counts(normalize=True); b=o.groupby(col).w.sum()/o.w.sum()
    t=pd.DataFrame({'uniform%':a*100,'EXP1_eff%':b*100}); t['ratio']=t['EXP1_eff%']/t['uniform%']; return t.round(2)
print('\n== Class distribution (objects) uniform vs EXP-1 effective'); print(comp('class_name'))
print('\n== Size'); print(comp('size_cat'))
print('\n== Taxonomy'); print(comp('taxonomy'))
print('\n== Mean D_i by class'); print(o.groupby('class_name').difficulty_score.agg(['mean','median','count']).round(3))
print('\n== Mean D_i by taxonomy'); print(o.groupby('taxonomy').difficulty_score.agg(['mean','count']).round(3))
# image level features
cnt=o.groupby('image_id').size(); m['nobj']=m.image_id.map(cnt).fillna(0)
print('Spearman D_image vs #objects: %.3f'%spearman(m.nobj.values,m.d_image.values))
print('mean #obj: all %.2f | hard %.2f | easy25 %.2f'%(m.nobj.mean(),m.nobj[hard].mean(),m.nobj[easy].mean()))
# primary class per image (majority)
maj=o.groupby('image_id').class_name.agg(lambda s:s.value_counts().index[0]); m['maj']=m.image_id.map(maj).fillna('BG')
t=m.groupby('maj').agg(n=('d_image','size'),meanD=('d_image','mean'),mass=('p_sampling','sum')); t['mass_ratio']=t.mass/(t.n/N); print(t.round(3))
# groups
rows=[]
for _,r_ in g[g.new_split=='train'].iterrows():
    for f in str(r_.member_filenames).split(';'):
        f=f.strip()
        if f: rows.append((f,r_.clean_group_id))
gm=pd.DataFrame(rows,columns=['image_id','gid']); m2=m.merge(gm,on='image_id',how='left')
print('\nmapped to group:',m2.gid.notna().sum(),'groups',m2.gid.nunique())
gs=m2.groupby('gid').size(); print('group size dist:',gs.value_counts().sort_index().to_dict())
multi=m2[m2.gid.map(gs)>1]
# within-group D std vs overall
wstd=multi.groupby('gid').d_image.std().mean(); print('mean within-group std D: %.3f | overall std D: %.3f'%(wstd,m.d_image.std()))
# ICC-ish: correlation of D between pairs
pairs=multi.groupby('gid').d_image.apply(list); a=[];b=[]
for L in pairs:
    if len(L)>=2: a.append(L[0]); b.append(L[1])
print('D correlation between members of same group (pearson): %.3f (n=%d)'%(np.corrcoef(a,b)[0,1],len(a)))
