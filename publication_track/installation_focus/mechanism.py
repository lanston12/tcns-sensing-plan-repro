"""Finite exact costs and conservative compatibility layer; no cache-log access."""
import itertools as it
import numpy as np
from publication_track.radio.protocol import BOOK,GRANTS,FALLBACK,QS,S

ACTIONS=BOOK[GRANTS,:,0]
INDEX={tuple(a):k for k,a in enumerate(ACTIONS)}
MIX=np.array([[[INDEX[tuple(np.where(z,b,a))] for z in QS] for a in ACTIONS] for b in ACTIONS])
COUNT=(ACTIONS>=0).sum(axis=1)
INFO=np.zeros((64,8,3))
for k,aa in enumerate(ACTIONS):
    for i,j in enumerate(aa):
        if j>=0:INFO[k,:,j]+=1/(S[i,j]*(1+3*QS[:,i]))

def grant_costs(p,price,scale):
    return (1/(1/np.asarray(p)+INFO/scale)).sum(axis=2)+price*COUNT[:,None]

def map_costs(p,maps,price,scale):
    maps=np.asarray(maps);precision=np.broadcast_to(1/np.asarray(p),(len(maps),8,3)).copy()
    actions=maps[:,np.arange(3)[None,:],QS]
    for i in range(3):
        for j in range(3):precision[:,:,j]+=(actions[:,:,i]==j)/(S[i,j]*scale*(1+3*QS[:,i]))
    return (1/precision).sum(axis=2)+price*(actions>=0).sum(axis=2)

def reachable_actions(versions,echoes,t):
    # An echo proves receipt only after it arrives. Installed versions increase.
    # Expiration fallback is deliberately always included, even if impossible now.
    sets=[]
    for i,echo in enumerate(echoes):
        values={int(BOOK[FALLBACK,i,0])}
        for v,(g,e,c) in versions.items():
            if v>=echo and t<e:values.add(int(BOOK[c,i,0]))
        sets.append(sorted(values))
    return sets

def choose(p,price,scale,last_code,versions,echoes,t,epsilon=.0005,compatible=True):
    c=grant_costs(p,price,scale);nom=c.mean(axis=1);best=nom.min()
    old=INDEX[tuple(BOOK[last_code,:,0])]
    host=old if nom[old]<=best+1e-12 else int(nom.argmin())
    sets=reachable_actions(versions,echoes,t)
    olds=np.array([INDEX[a] for a in it.product(*sets)])
    legal=np.flatnonzero(nom<=best+epsilon+1e-12)
    risk=(np.maximum(c[MIX[:,olds]]-np.maximum(c[olds][None,:,None,:],c[:,None,None,:]),0).max(axis=(1,2,3))
          if compatible else np.full(64,np.nan))
    pick=host
    if compatible:
        minimum=risk[legal].min();ties=legal[risk[legal]<=minimum+1e-12]
        # First keep the old complete plan on risk ties, then the host, then cheapest.
        pick=old if old in ties else host if host in ties else int(ties[np.argmin(nom[ties])])
    return int(GRANTS[pick]),dict(host_code=int(GRANTS[host]),selected_code=int(GRANTS[pick]),
        nominal_best=float(best),nominal_selected=float(nom[pick]),risk_host=float(risk[host]),
        risk_selected=float(risk[pick]),candidate_count=len(legal),old_vector_count=len(olds),
        reachable_actions=sets,echoes=list(echoes),context=list(p),epsilon=epsilon)

def jensen(p,price,scale,service=.75):
    # Exact reference for a DECLARED independent Bernoulli forward-service model.
    # These are not empirical NR masks and not a claim of service calibration.
    weights=np.prod(np.where(QS,service,1-service),axis=1)
    exact=np.zeros(64);mean_info=np.zeros((64,3))
    for z,w in zip(QS,weights):
        info=np.zeros((64,8,3))
        for i in range(3):
            for j in range(3):info[:,:,j]+=(ACTIONS[:,i,None]==j)*z[i]/(S[i,j]*scale*(1+3*QS[:,i]))
        exact+=w*(1/(1/np.asarray(p)+info)).sum(axis=2).mean(axis=1)
        mean_info+=w*info.mean(axis=1)
    exact+=price*COUNT
    closure=(1/(1/np.asarray(p)+mean_info)).sum(axis=1)+price*COUNT
    return exact,closure
