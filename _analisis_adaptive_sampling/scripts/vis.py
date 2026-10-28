import json, os
from PIL import Image, ImageDraw, ImageFont
R=os.path.expanduser('~/mnt/SKRIPSI'); d=json.load(open('vis_sel.json'))
NM=['CBB','CBSD','CGM','CMD','HEALTHY']; COL=[(255,60,60),(255,170,0),(0,200,255),(200,0,255),(0,255,90)]
try: F=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',15); FT=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',20)
except: F=FT=ImageFont.load_default()
preds={}
for r in d['preds']: preds.setdefault(r[0],[]).append(r[1:])
tiles=[]
for grp in ['hard','easy']:
  for f in d[grp][:8 if grp=='hard' else 4]:
    im=Image.open(f'{R}/cassava-v7-standardized/train/images/{f}').convert('RGB'); W,H=im.size; dr=ImageDraw.Draw(im)
    L=[l.split() for l in open(f'{R}/cassava-v7-standardized/train/labels/{os.path.splitext(f)[0]}.txt').read().split('\n') if l.strip()]
    for l in L:
      c=int(l[0]); xc,yc,w,h=map(float,l[1:5]); b=[(xc-w/2)*W,(yc-h/2)*H,(xc+w/2)*W,(yc+h/2)*H]
      dr.rectangle(b,outline=COL[c],width=5); dr.text((b[0]+4,b[1]+3),'GT '+NM[c],fill=COL[c],font=F,stroke_width=2,stroke_fill=(0,0,0))
    for c,cf,x1,y1,x2,y2 in preds.get(f,[]):
      c=int(c)
      for k in range(0,int(x2-x1),14): dr.line([(x1+k,y1),(min(x1+k+7,x2),y1)],fill=(255,255,255),width=2); dr.line([(x1+k,y2),(min(x1+k+7,x2),y2)],fill=(255,255,255),width=2)
      for k in range(0,int(y2-y1),14): dr.line([(x1,y1+k),(x1,min(y1+k+7,y2))],fill=(255,255,255),width=2); dr.line([(x2,y1+k),(x2,min(y1+k+7,y2))],fill=(255,255,255),width=2)
      dr.text((x1+4,y2-20),f'pred {NM[c]} {cf:.2f}',fill=(255,255,255),font=F,stroke_width=2,stroke_fill=(0,0,0))
    hdr=Image.new('RGB',(W,34),(20,20,20)); hd=ImageDraw.Draw(hdr)
    hd.text((8,6),f"{'SULIT' if grp=='hard' else 'MUDAH'}  D_image={d['dimg'][f]:.2f}  GT={len(L)}  pred>=0.25={len(preds.get(f,[]))}",fill=(255,255,255),font=FT)
    t=Image.new('RGB',(W,H+34)); t.paste(hdr,(0,0)); t.paste(im,(0,34)); tiles.append(t.resize((480,int((H+34)*480/W))))
cols=4; tw,th=tiles[0].size; rows=(len(tiles)+cols-1)//cols
M=Image.new('RGB',(cols*tw+(cols-1)*6,rows*th+(rows-1)*6),(255,255,255))
for i,t in enumerate(tiles): M.paste(t,((i%cols)*(tw+6),(i//cols)*(th+6)))
out=f'{R}/_analisis_adaptive_sampling/sampel_citra_sulit_vs_mudah.jpg'; M.save(out,quality=88); print(out,M.size)
