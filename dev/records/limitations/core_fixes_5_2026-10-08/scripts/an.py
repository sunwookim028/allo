import sys
rows=[l.split() for l in open(sys.argv[1])]
rows=[(int(a),int(b),int(c),int(d),int(e),int(f),int(g,16),int(h,16)) for a,b,c,d,e,f,g,h in rows]
mem={}
n_old=n_new=n_same=0; ex=[]
for i,(cyc,k,rst,we,wa,ra,wd,q) in enumerate(rows[:-1]):
    qn=rows[i+1][7]
    if we and wa==ra:
        old=mem.get(wa)
        if old is not None and old==wd: n_same+=1
        elif qn==wd: n_new+=1; ex.append((cyc,k,rst,wa,'new'))
        elif old is not None and qn==old: n_old+=1
        else: ex.append((cyc,k,rst,wa,'?',hex(qn),hex(wd),old and hex(old)))
    if we: mem[wa]=wd
print("collisions: old",n_old,"new",n_new,"same",n_same); print(ex[:12])
