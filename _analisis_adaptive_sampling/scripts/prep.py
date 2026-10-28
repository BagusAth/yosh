import pandas as pd, numpy as np, os, time
t0=time.time(); R=os.path.expanduser('~/mnt/SKRIPSI')
P=pd.read_csv(f'{R}/stage_8_1_oof_clean_run_20261020/oof_predictions_raw.csv',usecols=['image_id','class_id','confidence','x1','y1','x2','y2'],
  dtype={'image_id':'category','class_id':np.int8,'confidence':np.float32,'x1':np.float32,'y1':np.float32,'x2':np.float32,'y2':np.float32},engine='c')
print('read',round(time.time()-t0,1)); P.to_pickle('/tmp/ana/P.pkl'); print('saved',round(time.time()-t0,1),len(P))
