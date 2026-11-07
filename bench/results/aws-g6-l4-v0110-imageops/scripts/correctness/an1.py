import json, os, re, collections, sys
rows=[json.loads(l) for l in open(os.path.expanduser('~/results/trimref_trim.jsonl'))]
c=collections.Counter(); ex=collections.defaultdict(list)
for r in rows:
    im=r.get('im'); k=r.get('kept'); m=r['mode']; f=r['fuzz']; n=os.path.basename(r['img'])
    if not k: c[('nokept',m)]+=1; continue
    w=None
    hit=[i for i in r['issues'] if i.startswith('size differs from magick')]
    if not hit: c[('same',m)]+=1; continue
    if im is None: cls='im none'
    elif im[2]==0 or im[3]==0: cls='im 0-size: '+('1x1/line kept' if k[0]==1 or k[1]==1 else 'other')
    else: cls='im has box, sizes differ'
    c[(cls,m)]+=1
    ex[(cls,m)].append((n,f,k,im,r.get('doc')))
for k,v in sorted(c.items(),key=str): print(v,k)
for k,v in ex.items():
    print(k,len(v)); 
    for e in v[:14]: print('   ',e)
print("=====")
for m in ('ll','444'):
    for e in sorted(ex[('im has box, sizes differ',m)],key=str): print(m,e)
