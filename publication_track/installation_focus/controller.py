"""Fixed-host layer and a paid, fallible prepare/commit diagnostic."""
import json,time
from pathlib import Path
import numpy as np
from publication_track.radio.controller import Controller as Base
from publication_track.radio.protocol import BOOK,FALLBACK,encode,decode
from simulation.semantic.v5_continuous import DATA,write_csv
from .mechanism import choose

class Controller(Base):
    def __init__(self,cfg,obs):
        self.family=cfg['policy'];base=dict(cfg)
        base['policy']={'OLD_RENEWAL':'PRESCRIPTION'}.get(self.family,self.family)
        super().__init__(base,obs)
        self.echoes=[0]*3;self.queued={};self.selection=[];self.timing=[]
        self.prepared=[None]*3;self.transaction=None;self.acknowledged=set();self.protocol_events=[]

    def feedback_decision(self,t):
        started=time.perf_counter()
        if self.family in ('COV_EVENT','OLD_RENEWAL'):
            raw=super().feedback_decision(t)
        else:
            raw=None;due=t-self.last_fb>=self.cfg['period_s']-1e-6
            if self.family=='PHASED' and self.transaction is not None:
                v,g,e,c=self.transaction
                if t>=e:
                    self.protocol_events.append(dict(time=t,event='sender_timeout',version=v,node=3));self.transaction=None
                elif len(self.acknowledged)==3:
                    raw=encode(v,g,e,c,3);self.transaction=None
                    self.protocol_events.append(dict(time=t,event='commit_sent',version=v,node=3))
                due=False
            if due:
                code,detail=choose(self.c,self.price,self.scale,self.last_code,self.code_versions,self.echoes,t,
                    self.cfg['epsilon'],compatible=self.family=='COMPAT')
                detail.update(tick=self.k,time_s=t);self.selection.append(detail)
                self.version+=1;self.last_fb=t;self.last_code=code;e=t+self.cfg['ttl']
                mode=2 if self.family=='PHASED' else 0
                raw=encode(self.version,t,e,code,mode)
                self.code_versions[self.version]=(t,e,code)
                if mode==2:self.transaction=(self.version,t,e,code);self.acknowledged=set()
            self.feedback.append(dict(tick=self.k,time_s=t,send=int(raw is not None),code=self.last_code,
                lifetime=self.cfg['ttl'],version=self.version))
        elapsed=time.perf_counter()-started
        self.timing.append(dict(tick=self.k,decision_s=elapsed,send=int(raw is not None),
            modeled_processing_s=self.cfg['processing_s'],exceeds_budget=int(elapsed>self.cfg['processing_s'])))
        return raw

    def drain(self,t):
        super().drain(t)
        if self.family!='PHASED':return
        for i,p in enumerate(self.prepared):
            if p is None:continue
            v,g,e,c,committed=p
            if t>=e:
                self.protocol_events.append(dict(time=t,event='prepare_expired',version=v,node=i));self.prepared[i]=None
            elif committed and t>=g+.2:
                if v>self.code_cache[i][0]:self.code_cache[i]=(v,g,e,c)
                self.protocol_events.append(dict(time=t,event='activated',version=v,node=i));self.prepared[i]=None

    def send(self,node,pid,t):
        if pid>=2000000:
            return self.queued.pop(pid-2000000,None)
        if node==3 and pid<1000000:
            self.advance(pid//10,t);self.queued[pid//10]=self.feedback_decision(t);return None
        raw=super().send(node,pid,t)
        if self.family=='PHASED' and node<3 and pid<1000000 and raw is not None:
            p=self.prepared[node]
            if p is not None:
                values=list(DATA.unpack(raw));values[4]=32768+p[0];raw=DATA.pack(*values)
        return raw

    def handle(self,line):
        response=super().handle(line);a=line.split()
        if a[0]=='S' and int(a[2])>=2000000 and response!='-':
            raw=bytes.fromhex(response);v,g,e,c,mode,p=decode(raw)
            self.sent[int(a[2])]['generation_s']=f'{g:.6f}'
        return response

    def receive(self,node,pid,t,raw):
        if self.family=='PHASED' and raw[0]==3 and node<3:
            if (pid,node) in self.applied:return
            self.applied.add((pid,node));v,g,e,c,mode,p=decode(raw);delay=t-g
            self.rx.append(dict(packet_id=pid,node=node,kind='feedback',generation_s=g,radio_arrival_s=t,
                radio_delay_ms=delay*1000,app_erased=0,added_delay_ms=0,deadline_delivery=int(delay<=.15)))
            pending=self.prepared[node];event='rejected'
            if t>=e:event='expired'
            elif v<=self.code_cache[node][0]:event='obsolete'
            elif mode==2:
                if pending is None or v>pending[0]:self.prepared[node]=(v,g,e,c,False);event='prepared'
                else:event='duplicate_or_obsolete_prepare'
            elif mode==3:
                if pending is not None and pending[:4]==(v,g,e,c):
                    self.prepared[node]=(v,g,e,c,True);event='committed'
                else:event='commit_without_matching_prepare'
            self.protocol_events.append(dict(time=t,event=event,version=v,node=node));self.drain(t);return
        # Prepared acknowledgments share the already paid two-byte echo field.
        # Base only uses echoes to condition the old renewal model; high bits cannot
        # change the new chooser's information set or the raw observation filter.
        duplicate=(pid,node) in self.applied
        super().receive(node,pid,t,raw)
        if not duplicate and node==3 and raw[0]==5:
            _,j,i,ms,v,z,r=DATA.unpack(raw)
            if t-ms/1000<=.15+1e-9:
                if v>=32768:
                    if self.transaction is not None and v-32768==self.transaction[0]:self.acknowledged.add(i)
                else:self.echoes[i]=max(self.echoes[i],v)

    def save(self,folder):
        self.runtime=[r['decision_s'] for r in self.timing]
        super().save(folder);folder=Path(folder)
        write_csv(folder/'selection.csv',[{k:json.dumps(v) if isinstance(v,list) else v for k,v in d.items()} for d in self.selection])
        write_csv(folder/'timing.csv',self.timing);write_csv(folder/'protocol_events.csv',self.protocol_events)
        times=[r['decision_s'] for r in self.timing]
        (folder/'controller_metrics.json').write_text(json.dumps(dict(runtime_mean_s=float(np.mean(times)),
            runtime_p95_s=float(np.percentile(times,95)),expired_receptions=self.expirations,
            processing_s=self.cfg['processing_s'],epsilon=self.cfg['epsilon'],
            compute_exceeds_budget=sum(r['exceeds_budget'] for r in self.timing))))
