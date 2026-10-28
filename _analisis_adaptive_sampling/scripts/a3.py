import pandas as pd, numpy as np, os, glob
R=os.path.expanduser('~/mnt/SKRIPSI')
o=pd.read_csv(f'{R}/stage_8_2_difficulty_clean_20261020/train_object_difficulty.csv')
m=pd.read_csv(f'{R}/stage_8_2_difficulty_clean_20261020/train_image_difficulty_manifest.csv')
X=pd.concat([pd.read_pickle(f) for f in glob.glob('shard_*.pkl')])
o=o.merge(X,on=['image_id','gt_id'],how='left'); print('rows',len(o),'nan',o.best_same_conf50.isna().sum())
o.to_pickle('obj_validity.pkl')
pd.set_option('display.width',200)
print('\nCandidates per GT with IoU>=0.10: median %d, mean %.1f'%(o.n_cand_iou10.median(),o.n_cand_iou10.mean()))
print('Matched confidence by taxonomy (median / share conf<0.05):')
print(o[o.match_status=='Matched'].groupby('taxonomy').confidence.agg(median='median',lt005=lambda s:(s<0.05).mean()).round(3))
T=0.25
def status(r):
    if r.best_same_conf50>=T: return '1_Detected_OK'
    if r.best_any_conf50>=T and r.best_any_cls!=r.class_id: return '2_Confused_real'
    if r.best_same_conf50>=0.05: return '3_Weak_correct'
    return '4_Not_found'
C={'CBB':0,'CBSD':1,'CGM':2,'CMD':3,'HEALTHY':4}; o['class_id']=o.class_name.map(C)
o['real']=o.apply(status,axis=1)
print('\n== Pipeline taxonomy (rows) vs real status @conf>=0.25, IoU>=0.5 (row %)')
ct=pd.crosstab(o.taxonomy,o.real,normalize='index')*100; ct['n']=o.taxonomy.value_counts(); print(ct.round(1))
cc=o[o.taxonomy=='Classification_Confusion']
print('\nClassification_Confusion n=%d: correct-class box IoU>=.5 exists with HIGHER conf than matched wrong box: %.1f%%'%(len(cc),(cc.best_same_conf50>cc.confidence).mean()*100))
print('  of which the correct box has conf>=0.25: %.1f%%'%((cc.best_same_conf50>=T).mean()*100))
tp=o[o.taxonomy=='True_Positive']
print('TP n=%d: matched conf < best same-class conf (D inflated by low-conf duplicate): %.1f%%; median matched %.3f vs median best %.3f'%(len(tp),(tp.confidence<tp.best_same_conf50-1e-6).mean()*100,tp.confidence.median(),tp.best_same_conf50.median()))
ok=o.real=='1_Detected_OK'
print('\nObjects actually detected OK: %d (%.1f%%); of these with D_i>=0.70: %.1f%%; mean D_i %.3f'%(ok.sum(),ok.mean()*100,(o[ok].difficulty_score>=0.7).mean()*100,o[ok].difficulty_score.mean()))
print('Real status share overall:'); print((o.real.value_counts(normalize=True)*100).round(1).sort_index())
# image-level
hard=set(m.image_id[m.d_image>=0.8]); oh=o[o.image_id.isin(hard)]
print('\nIn HARD images (D_image>=0.8): objects n=%d, real status:'%len(oh)); print((oh.real.value_counts(normalize=True)*100).round(1).sort_index())
# real confusion pairs
rc=o[o.real=='2_Confused_real']; inv={v:k for k,v in C.items()}
print('\nReal confusion pairs (GT -> predicted), top:'); print(rc.groupby(['class_name',rc.best_any_cls.map(inv)]).size().sort_values(ascending=False).head(10))
# pipeline confusion pairs: need matched class - not stored; skip
# D_i vs real status
print('\nMean D_i by real status:'); print(o.groupby('real').difficulty_score.agg(['mean','count']).round(3))
# correlation D_i with (1-conf) of best same
print('corr(D_i, 1-best_same_conf): %.3f ; corr(D_i, 1-matched_conf): %.3f'%(np.corrcoef(o.difficulty_score,1-o.best_same_conf50)[0,1],np.corrcoef(o.difficulty_score,1-o.confidence)[0,1]))
