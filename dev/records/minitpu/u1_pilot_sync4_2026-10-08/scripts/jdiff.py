import sys, xml.etree.ElementTree as ET
def load(p):
    r={}
    for tc in ET.parse(p).iter('testcase'):
        k=tc.get('classname')+'::'+tc.get('name')
        st='pass'
        for c in tc:
            if c.tag in('failure','error'): st='fail'; break
            if c.tag=='skipped': st='skip'
        r[k]=st
    return r
a,b=load(sys.argv[1]),load(sys.argv[2])
from collections import Counter
print('base',Counter(a.values()),'branch',Counter(b.values()))
for k in sorted(set(a)|set(b)):
    x,y=a.get(k,'absent'),b.get(k,'absent')
    if x!=y and not (x=='absent' and y=='pass'): print(f'{x:>6} -> {y:<6} {k}')
print('new-and-passing on branch:',sum(1 for k in b if k not in a and b[k]=='pass'))
