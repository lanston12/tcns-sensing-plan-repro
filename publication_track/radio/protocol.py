"""Shared finite prescription class, byte codec, and joint posterior objective."""
import itertools as it, struct, zlib
import numpy as np

S=np.array([[.01,.04,.06],[.06,.01,.04],[.04,.06,.01]])
QS=np.array(list(it.product(range(2),repeat=3)))
MASKS=QS.copy()
HEADER=struct.Struct('<BBHIIHB') # kind, book, version, generation ms, expiry ms, id, mode

def make_book():
    # All 64 grants plus local-quality mappings derived from each grant.
    grants=np.array([[[a,a] for a in aa] for aa in it.product(range(-1,3),repeat=3)])
    adaptive=[]
    for pi in grants:
        # Bad-quality sources either abstain, or use their individually best public sensor.
        for mode in [0,1]:
            p=pi.copy()
            for i in range(3):p[i,1]=-1 if mode==0 else int(np.argmin(S[i]))
            adaptive.append(p)
    book=np.unique(np.concatenate([grants,np.array(adaptive)]),axis=0)
    return book,np.flatnonzero((book[:,:,0]==book[:,:,1]).all(axis=1))

BOOK,GRANTS=make_book()
FALLBACK=int(np.flatnonzero((BOOK==np.array([[0,0],[1,1],[2,2]])).all(axis=(1,2)))[0])

def evaluate(p,pis=BOOK,price=.003,scale=1.):
    pis=np.asarray(pis);a=pis[:,np.arange(3)[None,:],QS[:, :]] # K x 8 x 3
    precision=np.broadcast_to(1/np.asarray(p),(len(pis),8,3)).copy()
    for i in range(3):
        for j in range(3):precision[:,:,j]+=(a[:,:,i]==j)/(S[i,j]*scale*(1+3*QS[:,i]))
    return (1/precision).sum(axis=2).mean(axis=1)+price*(a>=0).sum(axis=2).mean(axis=1)

def encode(version,gen,expiry,code,mode=0,p=None,bits=8):
    head=HEADER.pack(3,1,version,round(gen*1000),round(expiry*1000),code,mode)
    if p is not None:
        # Three scalar log-uniform reconstruction indices, each stored in 1 or 2 bytes.
        bins=2**bits-1;v=np.rint(np.clip((np.log(np.maximum(p,1e-4))-np.log(1e-4))/np.log(3000),0,1)*bins).astype(int)
        head+=bytes([bits])+struct.pack('<'+('B' if bits<=8 else 'H')*3,*v)
    return head+struct.pack('<I',zlib.crc32(head))

def decode(raw):
    if len(raw)<HEADER.size+4 or zlib.crc32(raw[:-4])!=struct.unpack('<I',raw[-4:])[0]:raise ValueError('integrity')
    kind,book,v,g,e,c,mode=HEADER.unpack(raw[:HEADER.size])
    if kind!=3 or book!=1 or c>=len(BOOK):raise ValueError('unsupported codebook')
    p=None
    if mode==1:
        bits=raw[HEADER.size];fmt='<'+('B' if bits<=8 else 'H')*3
        q=np.array(struct.unpack(fmt,raw[HEADER.size+1:-4]));p=1e-4*np.exp(q/(2**bits-1)*np.log(3000))
    return v,g/1000,e/1000,c,mode,p
