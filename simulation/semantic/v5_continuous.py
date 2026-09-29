"""V5 causal controller. The same event stream can be replayed without ns-3.

No decision function reads truth, future observations, or radio receive logs at
another node. Socket callbacks consume simulated time, not wall-clock time.
"""
from __future__ import annotations
import csv, hashlib, heapq, json, math, random, struct
from pathlib import Path
from simulation.semantic.receiver_summary_v1 import ReceiverSummary, encode_summary, decode_summary

Q = .008
PRIOR = .04
S = ((.01,.04,.06),(.06,.01,.04),(.04,.06,.01))
DATA = struct.Struct('<BBHIHff') # format 5, target, source, generation ms, echo version, raw z, variance

def write_csv(path, rows):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows: return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)

def generate(seed, ticks=200):
    rng=random.Random(seed); x=[rng.gauss(0,1) for _ in range(3)]; rows=[]
    # Independent observation/visibility seed; no policy can branch on regimes.
    regimes=[rng.randrange(3) for _ in range((ticks+19)//20)]
    for k in range(ticks):
        for j in range(3):
            x[j]+=rng.gauss(0,math.sqrt(Q))
            r=.3 if j==regimes[k//20] else .004
            if k<20: r=.02  # fixed decision-invariant introductory interval
            rows.append(dict(tick=k,target=j,truth=x[j],r=r,
                local=x[j]+rng.gauss(0,math.sqrt(r)),
                **{f'z{i}':x[j]+rng.gauss(0,math.sqrt(S[i][j])) for i in range(3)}))
    return rows

def action(c,i,price):
    gains=[p*p/(p+S[i][j])-price for j,p in enumerate(c)]
    j=max(range(3),key=lambda j:gains[j])
    return j if gains[j]>0 else -1

def cost(a,c,i,price):
    return sum(c) if a<0 else sum(c)-c[a]**2/(c[a]+S[i][a])+price

class Controller:
    def __init__(self,cfg,observations):
        self.cfg=cfg; self.obs={(int(r['tick']),int(r['target'])):r for r in observations}
        self.policy=cfg['policy'];self.price=cfg['price'];self.k=-1
        self.local_p=[1.]*3;self.local_m=[0.]*3;self.c=[PRIOR]*3
        self.local_history=[];self.remote={};self.cache=[(0,2.1,[PRIOR]*3) for _ in range(3)]
        self.versions={0:(2.1,[PRIOR]*3)};self.belief=[{0:1.} for _ in range(3)]
        self.version=0;self.last_fb=-100.;self.last_signature=tuple(action(self.c,i,self.price) for i in range(3))
        self.last_echo_time=[-1.]*3;self.pending=[];self.serial=0
        self.sent={};self.rx=[];self.decisions=[];self.filters=[];self.feedback=[];self.events=[]
        self.belief_log=[];self.applied=set(); self.now=2.1

    def aged(self,v,t):
        if v==0:return [PRIOR]*3
        if v not in self.versions: return [PRIOR]*3
        gen,c=self.versions[v]
        # Public prediction-only aging proxy; unknown future local updates excluded.
        return [min(.3,p+Q*max(0.,t-gen)/.1) for p in c]

    def advance(self,k,t):
        if k==self.k:return
        assert k==self.k+1
        self.k=k
        for j in range(3):
            row=self.obs[k,j];p=self.local_p[j]+Q;gain=p/(p+row['r'])
            self.local_m[j]+=gain*(row['local']-self.local_m[j]);self.local_p[j]=(1-gain)*p
        self.c=self.local_p[:]
        self.local_history.append(self.local_m[:])

    def queue(self,time,kind,data):
        self.serial+=1;heapq.heappush(self.pending,(time,self.serial,kind,data))

    def drain(self,t):
        while self.pending and self.pending[0][0]<=t+1e-8:
            time,_,kind,d=heapq.heappop(self.pending)
            if kind=='belief':
                v=d;prob=self.cfg.get('pilot_feedback_success',.8)
                for i in range(3):
                    b={key:weight*(1-prob) for key,weight in self.belief[i].items()}
                    b[v]=b.get(v,0)+prob
                    # recent four versions plus unknown bucket. Unknown uses worst regret.
                    old=sorted(key for key in b if key>=0)[:-4]
                    for key in old:b[-1]=b.get(-1,0)+b.pop(key)
                    self.belief[i]=b
            elif kind=='summary':
                i,pid,raw=d;s=decode_summary(raw);key=(pid,i)
                accepted=s.cache_version>self.cache[i][0]
                if accepted:self.cache[i]=(s.cache_version,s.generation_ms/1000,list(s.variances))
                self.events.append(dict(time_s=t,event='cache_apply' if accepted else 'obsolete_summary',
                    node=i,packet_id=pid,version=s.cache_version,eligible_s=time))

    def feedback_decision(self,t):
        signature=tuple(action(self.c,i,self.price) for i in range(3))
        # Continuous action-change extension of V4 receiver-push, without ACK repair.
        changed=signature!=self.last_signature
        self.last_signature=signature
        periodic=t-self.last_fb>=self.cfg.get('period_s',.5)-1e-6
        refresh=t-self.last_fb>=1.-1e-6
        regret=0.
        for i in range(3):
            best=cost(action(self.c,i,self.price),self.c,i,self.price)
            for v,w in self.belief[i].items():
                if v==-1:
                    g=max(cost(a,self.c,i,self.price)-best for a in range(-1,3))
                else:g=cost(action(self.aged(v,t),i,self.price),self.c,i,self.price)-best
                regret+=w*g
        proxy=5*self.cfg.get('pilot_feedback_success',.8)*regret
        fb_cost=250*2e-6+37*1e-7
        send={'NO_CONTEXT':False,'FREE_CONTEXT':False,'PERIODIC_TUNED':periodic,
              'DUMMY_PERIODIC':periodic,'EVENT_V4':changed,'EVENT_REFRESH':changed or refresh,
              'DECISION_CACHE_AWARE':proxy>fb_cost and t-self.last_fb>=.2-1e-6,
              'NEUTRAL':periodic}[self.policy]
        self.feedback.append(dict(tick=self.k,time_s=t,send=int(send),signature=str(signature),
             expected_proxy_regret=regret,horizon5_proxy=proxy,cost_proxy=fb_cost,version=self.version))
        if not send:return None
        self.version+=1;self.last_fb=t
        c=[PRIOR]*3 if self.policy=='DUMMY_PERIODIC' else self.c
        raw=encode_summary(ReceiverSummary(self.version,round(t*1000),tuple(c)))
        decoded=decode_summary(raw);self.versions[self.version]=(t,list(decoded.variances))
        self.queue(t+.1,'belief',self.version)
        return raw

    def send(self,node,pid,t):
        if pid>=1000000: return bytes([6])+bytes(self.cfg.get('background_bytes',400)-21)
        k=pid//10;self.advance(k,t)
        if node==3:return self.feedback_decision(t)
        v,gen,c=self.cache[node]
        if self.policy=='FREE_CONTEXT':c=self.c[:];v=65535;gen=t
        elif self.policy in ('NO_CONTEXT','DUMMY_PERIODIC','NEUTRAL'):c=[PRIOR]*3;v=0;gen=2.1
        elif v==0:c=[PRIOR]*3
        else:c=[min(.3,p+Q*max(0,t-gen)/.1) for p in c]
        a=action(c,node,self.price)
        if self.policy=='NEUTRAL':a=k%3
        best=action(self.c,node,self.price)
        b=self.belief[node]
        expected_v=max(b,key=b.get)
        self.decisions.append(dict(tick=k,time_s=t,source=node,action=a,cache_version=v,
            cache_age_ms=(t-gen)*1000,expected_version=expected_v,
            mismatch=int(v!=expected_v),hindsight_regret=cost(a,self.c,node,self.price)-cost(best,self.c,node,self.price),
            **{f'cache_p{j}':c[j] for j in range(3)},**{f'actual_c{j}':self.c[j] for j in range(3)}))
        self.belief_log.append(dict(tick=k,source=node,belief=json.dumps(b),echo_generation=self.last_echo_time[node]))
        if a<0:return None
        return DATA.pack(5,a,node,round(t*1000),v,self.obs[k,a][f'z{node}'],S[node][a])

    def receive(self,node,pid,t,raw):
        spec=self.sent[pid]; assert spec['payload_hex']==raw.hex()
        kind='feedback' if raw[0]==3 else 'data' if raw[0]==5 else 'background'
        intended=(kind=='feedback' and node<3) or (kind=='data' and node==3)
        if not intended:return
        if (pid,node) in self.applied:return
        self.applied.add((pid,node));gen=float(spec['generation_s']);delay=t-gen
        intervention=self.cfg.get('intervention','none');drop=False;extra=0.
        k=pid//10
        if kind=='feedback' and intervention!='none':
            # Exogenous common broadcast burst + receiver-specific hash; no policy identity.
            h=int(hashlib.sha256(f"{self.cfg['network_seed']}|{k}|{node}".encode()).hexdigest()[:8],16)/2**32
            common=int(hashlib.sha256(f"{self.cfg['network_seed']}|{k}|common".encode()).hexdigest()[:8],16)/2**32
            elapsed=gen-2.1
            drop=(common<.2 or h<.12)
            extra=.25 if h>.65 else 0.
            if intervention=='burst' and (5<=elapsed<9 or 13<=elapsed<15):drop=True
        self.rx.append(dict(packet_id=pid,node=node,kind=kind,generation_s=gen,radio_arrival_s=t,
            radio_delay_ms=delay*1000,app_erased=int(drop),added_delay_ms=extra*1000,
            deadline_delivery=int(not drop and delay+extra<=.15)))
        if drop:return
        if kind=='feedback':
            self.queue(t+extra,'summary',(node,pid,raw))
        else:
            if delay>.15:return # explicit common late raw observation discard
            _,j,i,ms,echo,z,var=DATA.unpack(raw);k=round((ms/1000-2.101)/.1)
            self.remote[k,i]=(j,z,var)
            # Echo refers to sender cache at data generation, not magically current cache.
            if gen>self.last_echo_time[i] and echo!=65535:
                self.last_echo_time[i]=gen
                b={echo:1.}
                for v,(vg,_) in self.versions.items():
                    if v and gen<vg+.1<=t:
                        prob=self.cfg.get('pilot_feedback_success',.8)
                        b={key:w*(1-prob) for key,w in b.items()};b[v]=b.get(v,0)+prob
                self.belief[i]=b
        self.drain(t)

    def score(self,t):
        # Refilter only observations that actually arrived by t; raw independent data.
        m=[0.]*3;p=[1.]*3
        for k in range(self.k+1):
            for j in range(3):
                p[j]+=Q;row=self.obs[k,j];g=p[j]/(p[j]+row['r'])
                m[j]+=g*(row['local']-m[j]);p[j]*=1-g
            for i in range(3):
                if (k,i) in self.remote:
                    j,z,r=self.remote[k,i];g=p[j]/(p[j]+r);m[j]+=g*(z-m[j]);p[j]*=1-g
        for j in range(3):
            truth=self.obs[self.k,j]['truth'] # evaluation only
            self.filters.append(dict(tick=self.k,time_s=t,target=j,truth=truth,estimate=m[j],
                covariance=p[j],local_estimate=self.local_m[j],local_covariance=self.c[j],
                observation_variance=self.obs[self.k,j]['r'],squared_error=(truth-m[j])**2,
                local_squared_error=(truth-self.local_m[j])**2))

    def handle(self,line):
        parts=line.split();kind=parts[0];t=float(parts[2] if kind=='T' else parts[3]);self.now=t;self.drain(t)
        if kind=='S':
            node,pid=int(parts[1]),int(parts[2]);raw=self.send(node,pid,t)
            if raw is None:return '-'
            self.sent[pid]=dict(source=node,packet_id=pid,generation_s=f'{t:.6f}',send_s=f'{t:.6f}',
                bytes=len(raw)+20,payload_hex=raw.hex())
            return raw.hex()
        if kind=='R':self.receive(int(parts[1]),int(parts[2]),t,bytes.fromhex(parts[4]))
        elif kind=='T':self.score(t)
        return 'OK'

    def save(self,folder):
        folder=Path(folder)
        for name,rows in [('schedule',list(self.sent.values())),('receives',self.rx),('decisions',self.decisions),
                          ('filter',self.filters),('feedback',self.feedback),('cache_events',self.events),('belief',self.belief_log)]:
            write_csv(folder/(name+'.csv'),rows)
