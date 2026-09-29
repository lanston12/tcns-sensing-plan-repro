"""Checks that can expose a wrong risk constraint, illegal cache use or handshake."""
import itertools as it,json,time
from pathlib import Path
import numpy as np
from publication_track.radio.protocol import BOOK,GRANTS,FALLBACK,encode
from publication_track.radio.controller import generate
from .mechanism import choose,map_costs,grant_costs,reachable_actions,jensen
from .controller import Controller

def main():
    rng=np.random.RandomState(37);maxerror=0;cases=0
    for _ in range(50):
        p=np.exp(rng.uniform(-6,-1,3));scale=rng.uniform(.6,1.8);price=.006
        costs=grant_costs(p,price,scale)
        maxerror=max(maxerror,float(np.max(abs(costs-map_costs(p,BOOK[GRANTS],price,scale)))))
        versions={0:(2.,2.,FALLBACK),1:(2.,3.,int(rng.choice(GRANTS))),2:(2.1,3.,int(rng.choice(GRANTS)))}
        code,d=choose(p,price,scale,versions[2][2],versions,[0,1,0],2.2)
        assert d['nominal_selected']<=d['nominal_best']+.0005+1e-12
        assert d['risk_selected']<=d['risk_host']+1e-12
        exact,closure=jensen(p,price,scale);assert np.min(exact-closure)>-1e-12;cases+=1
    assert maxerror<1e-12
    cfg=dict(policy='PHASED',observation_seed=1,price=.006,period_s=.2,ttl=.4,epsilon=.0005,processing_s=.04)
    c=Controller(cfg,generate(1,5));code=int(GRANTS[3])
    def rx(pid,node,t,v=1,mode=2):c.receive(node,pid,t,encode(v,2.1,2.5,code,mode))
    rx(1,0,2.15,mode=3);assert c.prepared[0] is None # COMMIT before PREPARE
    rx(2,0,2.16);assert c.code_cache[0][0]==0 # PREPARE alone never activates
    c.drain(2.31);assert c.code_cache[0][0]==0 # lost COMMIT stays old
    rx(3,1,2.17);rx(4,1,2.18,mode=3);assert c.code_cache[1][0]==0
    c.drain(2.31);assert c.code_cache[1][0]==1 and c.code_cache[0][0]==0 # non-atomic
    c.drain(2.51);assert c.prepared[0] is None # expiry
    rx(5,2,2.6);assert c.prepared[2] is None # expired PREPARE
    c.transaction=(1,2.1,2.5,code);c.acknowledged={0,1};c.k=0;c.c=[.01]*3
    assert c.feedback_decision(2.4) is None # lost ACK prevents commit
    assert c.feedback_decision(2.6) is None and c.transaction is None
    # Receiver selections cannot depend on actual remote caches or truth labels.
    c1=Controller(dict(cfg,policy='COMPAT'),generate(2,5));c2=Controller(dict(cfg,policy='COMPAT'),generate(2,5))
    c2.code_cache=[(91,2.,5.,int(GRANTS[0]))]*3
    for r in c2.obs.values():r['truth']+=1000
    c1.advance(0,2.1);c2.advance(0,2.1)
    assert c1.feedback_decision(2.1)==c2.feedback_decision(2.1)
    out=dict(random_cost_constraint_cases=cases,max_cost_crosscheck_error=maxerror,
        protocol_branches=['commit_without_prepare','lost_commit','lost_ACK','sender_timeout','expired_prepare','staged_activation_not_atomic'],
        causal_invariance=True,jensen_nonnegative=True)
    (Path(__file__).parent/'results/tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
