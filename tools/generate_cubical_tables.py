#!/usr/bin/env python3
"""Regenerate the package's 3D relative cubical F2 plans (stdlib only).

Each neighborhood is a downset of {-1,0,1}^3 under replacing coordinates
by zero. Reduce its relative tensor boundaries in global orientation/anchor
order. Store full square combinations, not just local pivot labels. No input
image or competitor implementation is used. --check verifies the shipped data.
"""
from pathlib import Path
import argparse, itertools, struct, hashlib
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--check',action='store_true')
args=p.parse_args()
data={};counts={}
for n in (3,):
 slots=list(itertools.product((-1,0,1),repeat=n));lookup={s:i for i,s in enumerate(slots)}
 center=lookup[(0,)*n];dims=[sum(x!=0 for x in s) for s in slots]
 deps=[]
 for s in slots:
  dep=0
  for a,x in enumerate(s):
   if x:
    t=list(s);t[a]=0;dep|=1<<lookup[tuple(t)]
  deps.append(dep)
 sequence=sorted((i for i in range(len(slots)) if i!=center),key=lambda i:(dims[i],i))
 patterns=[1<<center]
 for i in sequence:
  patterns += [m|(1<<i) for m in patterns if m&deps[i]==deps[i]]
 assert len(set(patterns))==len(patterns)
 for convention in ('V','T'):
  degree=dims if convention=='V' else [n-d for d in dims]
  def orderkey(i):
   s=slots[i];orientation=sum((1<<a) for a,x in enumerate(s) if (x!=0 if convention=='V' else x==0))
   anchor=tuple(min(x,0) if convention=='V' else int(x==1) for x in s)
   return (orientation,tuple(reversed(anchor)))
  cells=[sorted((i for i in range(len(slots)) if degree[i]==d),key=orderkey) for d in range(n+1)]
  boundaries=[0]*len(slots)
  for i,s in enumerate(slots):
   for a,x in enumerate(s):
    if convention=='V' and x:
     t=list(s);t[a]=0;boundaries[i]^=1<<lookup[tuple(t)]
    elif convention=='T' and x==0:
     for side in (-1,1):
      t=list(s);t[a]=side;boundaries[i]^=1<<lookup[tuple(t)]
  records=[];checks=0
  for mask in sorted(patterns):
   # Verify d^2=0 in the local relative complex for every admissible pattern.
   for d in range(2,n+1):
    for c in cells[d]:
     if not mask>>c&1:continue
     b=boundaries[c]&mask;bb=0
     while b:
      bit=b&-b;j=bit.bit_length()-1;b^=bit;bb^=boundaries[j]&mask
     assert bb==0;checks+=1
   pairsets=[];combos=[]
   for d in range(1,n+1):
    positions={c:i for i,c in enumerate(cells[d-1])};pivots={};neg=0;pos=0;pair=[]
    for c in cells[d]:
     if not mask>>c&1:continue
     col=sum(1<<positions[b] for b in cells[d-1] if (boundaries[c]&mask)>>b&1)
     combination=1<<c
     while col:
      pivot=col.bit_length()-1
      if pivot not in pivots:break
      old,chain=pivots[pivot];col^=old;combination^=chain
     if col:
      pivot=col.bit_length()-1;pivots[pivot]=(col,combination)
      row=cells[d-1][pivot];neg|=1<<c;pos|=1<<row;pair.append([row,combination])
    pairsets.append((neg,pos));combos.append(pair)
   assert not (pairsets[0][0]&pairsets[1][1])
   records.append([mask,pairsets[0][0],pairsets[1][1],pairsets[1][0],combos[1]])
  key=f'{n}{convention}'
  data[key]={'slots':slots,'center':center,'deps':deps,'sequence':sequence,'cells':cells,'boundaries':boundaries,'records':records}
  counts[key]={'patterns':len(patterns),'boundary_squared_checks':checks,'pairs':sum(len(x[-1]) for x in records)}

blob=bytearray(b'CT3D0001')
for key in ('3V','3T'):
 records=data[key]['records']
 blob.extend(struct.pack('<I',len(records)))
 for mask,negative_edges,positive_edges,negative_faces,pairs in records:
  blob.extend(struct.pack('<4I',mask,negative_edges,negative_faces,len(pairs)))
  for slot,chain in pairs:
   assert 0<=slot<27 and 0<=chain<(1<<27)
   blob.extend(struct.pack('<I',(chain<<5)|slot))
path=Path(__file__).resolve().parents[1]/'src/ordinary_persistence/cubical_tables.bin'
if args.check:
 assert path.read_bytes()==blob, 'cubical table data need regeneration'
else:
 path.write_bytes(blob)
print(counts)
print(len(blob), hashlib.sha256(blob).hexdigest())
