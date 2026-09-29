"""Publication controller reuses the V5 causal callback and raw-data format."""
import json, math, random, time
import numpy as np
from simulation.semantic.v5_continuous import Controller as Legacy, DATA, write_csv
from .protocol import BOOK,GRANTS,FALLBACK,QS,MASKS,S,evaluate,encode,decode

def generate(seed,ticks=200):
    rng=random.Random(seed);q=.004+rng.random()*.012;scale=.6+rng.random()*1.2
    x=[rng.gauss(0,1) for _ in range(3)];rows=[];bad=rng.randrange(3);quality=[0,0,0]
    transition=.025+rng.random()*.12
    for k in range(ticks):
        if rng.random()<transition:bad=rng.randrange(3)
        for i in range(3):
            if rng.random()<.15:quality[i]=1-quality[i]
        for j in range(3):
            x[j]+=rng.gauss(0,math.sqrt(q));r=(.18+rng.random()*.15) if j==bad else .0035
            row=dict(tick=k,target=j,truth=x[j],r=r,local=x[j]+rng.gauss(0,math.sqrt(r)),q=q,scale=scale)
            for i in range(3):
                var=float(S[i,j]*scale*(1+3*quality[i]));row.update({f'z{i}':x[j]+rng.gauss(0,math.sqrt(var)),f's{i}':var,f'w{i}':quality[i]})
            rows.append(row)
    return rows

class Controller(Legacy):
    def __init__(self,cfg,observations):
        super().__init__(cfg,observations)
        self.q=observations[0]['q'];self.scale=observations[0]['scale']
        self.code_cache=[(0,2.1,2.1,FALLBACK) for _ in range(3)]
        self.code_versions={0:(2.1,2.1,FALLBACK)}
        self.joint={(0,0,0):1.};self.pending_models=[];self.runtime=[];self.expirations=0
        self.masks=np.array(cfg.get('feedback_masks',[.25,.08,.08,.08,.08,.08,.08,.27]))
        self.masks=self.masks/self.masks.sum()
        self.last_code=FALLBACK;self.last_feedback_raw=None

    def refilter(self):
        m=np.zeros(3);p=np.ones(3)
        for k in range(self.k+1):
            p+=self.q
            for j in range(3):
                row=self.obs[k,j];g=p[j]/(p[j]+row['r']);m[j]+=g*(row['local']-m[j]);p[j]*=1-g
            for i in range(3):
                if (k,i) in self.remote:
                    j,z,r=self.remote[k,i];g=p[j]/(p[j]+r);m[j]+=g*(z-m[j]);p[j]*=1-g
        return m,p

    def advance(self,k,t):
        if k==self.k:return
        assert k==self.k+1;self.k=k
        for j in range(3):
            row=self.obs[k,j];p=self.local_p[j]+self.q;g=p/(p+row['r'])
            self.local_m[j]+=g*(row['local']-self.local_m[j]);self.local_p[j]=(1-g)*p
        self.c=self.refilter()[1].tolist()

    def active(self,v,t):
        gen,exp,c=self.code_versions.get(v,(0,0,FALLBACK));return c if t<exp else FALLBACK

    def drain(self,t):
        # Caches change only in receive(); model installation is a separate approximation.
        ready=[x for x in self.pending_models if x[0]<=t+1e-8]
        self.pending_models=[x for x in self.pending_models if x[0]>t+1e-8]
        for when,v in ready:
            new={}
            for old,weight in self.joint.items():
                for mask,prob in zip(MASKS,self.masks):
                    vv=tuple(v if mask[i] else old[i] for i in range(3));new[vv]=new.get(vv,0)+weight*prob
            new=dict(sorted(new.items(),key=lambda x:-x[1])[:32]);z=sum(new.values());self.joint={k:w/z for k,w in new.items()}

    def rollout(self,t,code,ttl,send):
        # Conditional deterministic covariance surrogate, explicit joint installation masks.
        # Current R is frozen over forecast, never read from future observations.
        horizon=10;r=np.array([self.obs[self.k,j]['r'] for j in range(3)])
        branches=[]
        for vv,w in self.joint.items():
            if send:
                for mask,prob in zip(MASKS,self.masks):branches.append((vv,mask,w*prob))
            else:branches.append((vv,(0,0,0),w))
        # Merge equal source prescription/expiry signatures before evaluating.
        merged={}
        for vv,mask,w in branches:
            signature=tuple(((code,t+ttl) if mask[i] else (self.code_versions[vv[i]][2],self.code_versions[vv[i]][1]))+(self.active(vv[i],t),) for i in range(3))
            merged[signature]=merged.get(signature,0)+w
        signatures=list(merged);weights=np.array(list(merged.values()));p=np.tile(self.c,(len(signatures),1));cost=0.
        # Expected precisions and counts computed exactly over finite local classes;
        # covariance of expected information is the explicitly declared approximation.
        for h in range(horizon):
            tm=t+h*.1;info=np.zeros_like(p);count=np.zeros(len(p))
            for i in range(3):
                ids=np.array([s[i][0] if tm<s[i][1] else FALLBACK for s in signatures])
                if send and h==0:ids=np.array([s[i][2] for s in signatures])
                for w in range(2):
                    a=BOOK[ids,i,w];count+=(a>=0)*.5
                    for j in range(3):info[:,j]+=(a==j)*.5/(S[i,j]*self.scale*(1+3*w))
            service=.5 if send and h==0 else .75
            p=1/(1/p+service*info)
            cost+=float(weights@(p.sum(axis=1)+self.price*count))
            p=1/(1/(p+self.q)+1/r)
        return cost+(.003 if send else 0)

    def feedback_decision(self,t):
        start=time.perf_counter();costs=evaluate(self.c,price=self.price,scale=self.scale)
        if self.policy=='GRANT':code=int(GRANTS[np.argmin(costs[GRANTS])])
        elif self.policy=='GOQ':
            reps=np.array(self.cfg['goq_representatives']);ids=np.argmin(np.array([evaluate(p,price=self.price,scale=self.scale) for p in reps]),axis=1)
            code=int(ids[np.argmin(costs[ids])])
        else:code=int(np.argmin(costs))
        period=self.cfg.get('period_s',.5);ttl=self.cfg.get('ttl',period*2)
        due=t-self.last_fb>=period-1e-6
        send=due
        if self.policy=='NO_FEEDBACK':send=False
        if self.policy=='COV_EVENT':send=(code!=self.last_code and t-self.last_fb>=.2-1e-6) or due
        if self.policy=='PRESCRIPTION':
            options=[(self.rollout(t,code,ttl,False),False,code,ttl)]
            if t-self.last_fb>=.2-1e-6:
                for cand in sorted(set([code,self.last_code])):
                    for life in [.2,.5,1.]:options.append((self.rollout(t,cand,life,True),True,cand,life))
            _,send,code,ttl=min(options,key=lambda x:x[0])
        if self.policy=='PRESCRIPTION_FIXED':send=due
        self.runtime.append(time.perf_counter()-start)
        self.feedback.append(dict(tick=self.k,time_s=t,send=int(send),code=code,lifetime=ttl,version=self.version))
        if not send:return None
        self.version+=1;self.last_fb=t;self.last_code=code
        mode=1 if self.policy.startswith('COV') else 0
        raw=encode(self.version,t,t+ttl,code,mode,np.array(self.c) if mode else None,self.cfg.get('bits',8))
        _,gen,exp,decoded,mode,p=decode(raw)
        if p is not None:decoded=int(np.argmin(evaluate(p,price=self.price,scale=self.scale)))
        self.code_versions[self.version]=(gen,exp,decoded);self.pending_models.append((t+.1,self.version));self.last_feedback_raw=raw
        return raw

    def send(self,node,pid,t):
        if pid>=1000000:return bytes([6])+bytes(self.cfg.get('background_bytes',400)-21)
        self.advance(pid//10,t)
        if node==3:return self.feedback_decision(t)
        v,gen,exp,code=self.code_cache[node]
        if t>=exp or self.policy=='NO_FEEDBACK':code=FALLBACK
        q=self.obs[self.k,0][f'w{node}'];a=int(BOOK[code,node,q])
        self.decisions.append(dict(tick=self.k,time_s=t,source=node,action=a,cache_version=v,code=code,
             expired=int(t>=exp),cache_age_ms=(t-gen)*1000,mismatch=0,hindsight_regret=0))
        if a<0:return None
        row=self.obs[self.k,a]
        return DATA.pack(5,a,node,round(t*1000),v,row[f'z{node}'],row[f's{node}'])

    def receive(self,node,pid,t,raw):
        spec=self.sent[pid];assert spec['payload_hex']==raw.hex()
        kind='feedback' if raw[0]==3 else 'data' if raw[0]==5 else 'background'
        if not ((kind=='feedback' and node<3) or (kind=='data' and node==3)):return
        if (pid,node) in self.applied:return
        self.applied.add((pid,node));gen=float(spec['generation_s']);delay=t-gen
        self.rx.append(dict(packet_id=pid,node=node,kind=kind,generation_s=gen,radio_arrival_s=t,radio_delay_ms=delay*1000,
                           app_erased=0,added_delay_ms=0,deadline_delivery=int(delay<=.15)))
        if kind=='feedback':
            v,g,e,c,mode,p=decode(raw)
            if v>self.code_cache[node][0] and t<e:
                if p is not None:c=int(np.argmin(evaluate(p,price=self.price,scale=self.scale)))
                self.code_cache[node]=(v,g,e,c);event='installed'
            else:event='expired_or_obsolete';self.expirations+=1
            self.events.append(dict(time_s=t,event=event,node=node,packet_id=pid,version=v,expiry=e))
        elif delay<=.15:
            _,j,i,ms,echo,z,var=DATA.unpack(raw);k=round((ms/1000-2.101)/.1);self.remote[k,i]=(j,z,var)
            # Conservative echo conditioning: newer model versions may postdate this echo.
            keep={vv:w for vv,w in self.joint.items() if vv[i]==echo or self.code_versions[vv[i]][0]>gen}
            if keep:
                z0=sum(keep.values());self.joint={vv:w/z0 for vv,w in keep.items()}
            self.c=self.refilter()[1].tolist()

    def score(self,t):
        m,p=self.refilter()
        for j in range(3):
            row=self.obs[self.k,j];truth=row['truth']
            self.filters.append(dict(tick=self.k,time_s=t,target=j,truth=truth,estimate=m[j],covariance=p[j],
                 local_estimate=self.local_m[j],local_covariance=self.local_p[j],squared_error=(truth-m[j])**2,
                 local_squared_error=(truth-self.local_m[j])**2))

    def save(self,folder):
        super().save(folder)
        from pathlib import Path
        (Path(folder)/'controller_metrics.json').write_text(json.dumps(dict(runtime_mean_s=float(np.mean(self.runtime)),
             runtime_p95_s=float(np.percentile(self.runtime,95)),expired_receptions=self.expirations)))
