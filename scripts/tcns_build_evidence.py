"""Read-only reanalysis of archived runs; output only TCNS tables and figures."""
import csv
import json
import hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'publication_track/installation_focus'
R, OUT, FIG = BASE/'results', ROOT/'results', ROOT/'figures'
ENVS = ['reliable', 'partial', 'contention']
POL = ['GRANT_KEEP', 'COV_EVENT', 'OLD_RENEWAL', 'COMPAT']
NAMES = dict(GRANT_KEEP='Grant/keep', COV_EVENT='Cov/event', OLD_RENEWAL='Renewal', COMPAT='Compat.')
sources = set()

def read(p):
    sources.add(p)
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write(name, rows):
    with (OUT/name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))

def save(fig, name):
    fig.savefig(FIG/(name+'.pdf'), bbox_inches='tight', pad_inches=.035)
    fig.savefig(FIG/(name+'.png'), dpi=200, bbox_inches='tight', pad_inches=.035)
    plt.close(fig)

def main():
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':8, 'axes.titlesize':8.5,
                         'axes.labelsize':8, 'legend.fontsize':7, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    old = read(R/'existing_summary.csv'); radio = read(R/'radio_results.csv')
    assert len(old) == 720 and len(radio) == 108
    core = [r for r in radio if r['policy'] in POL and float(r['processing_s']) == .04]
    assert len(core) == 96
    primary=[]
    for e in ENVS:
        for p in POL:
            rr=[r for r in core if r['environment']==e and r['policy']==p]
            assert len(rr)==8
            primary.append(dict(environment=e,policy=p,runs=len(rr),mse=mean(rr,'mse'),
                                pssch_rb_slots=mean(rr,'pssch_tx_prb_slots')))
    write('tcns_primary_means.csv', primary)
    pairs=[]
    for c in core:
        if c['policy']!='COMPAT': continue
        h=next(h for h in core if h['policy']=='GRANT_KEEP' and all(h[k]==c[k] for k in ['environment','observation_seed','network_seed']))
        pairs.append(dict(environment=c['environment'],instance=c['observation_seed'],radio_seed=c['network_seed'],
                          host_tag=h['tag'],compatibility_tag=c['tag'],host_mse=float(h['mse']),compat_mse=float(c['mse']),
                          mse_delta=float(c['mse'])-float(h['mse']),
                          resource_delta=float(c['pssch_tx_prb_slots'])-float(h['pssch_tx_prb_slots'])))
    write('tcns_all_24_pairs.csv', pairs)
    assert len(pairs)==24 and sum(p['mse_delta']==0 and p['resource_delta']==0 for p in pairs)==20
    groups=[]
    policies=['NO_FEEDBACK','COV_PERIODIC','COV_EVENT','GRANT','GOQ','PRESCRIPTION']
    for e in ENVS:
        for p in policies:
            rr=[r for r in old if r['environment']==e and r['policy']==p]
            updates=sum(int(r['updates']) for r in rr)
            groups.append(dict(environment=e,policy=p,runs=len(rr),updates=updates,
                identical_updates=sum(int(r['identical_updates']) for r in rr),
                strict_improvements=sum(int(r['strict_improvements']) for r in rr),
                possible_harmful_updates=sum(int(r['harmful_possible_updates']) for r in rr),
                harmful_seconds=mean(rr,'harmful_seconds'),latest_two_seconds=mean(rr,'harmful_within_latest_two_seconds'),
                outside_latest_two_seconds=mean(rr,'harmful_seconds')-mean(rr,'harmful_within_latest_two_seconds'),
                expired_source_seconds=mean(rr,'expired_source_seconds')))
    write('tcns_mechanism_prevalence.csv', groups)
    # Expiration is a cross-tabulation, not a disjoint explanation of trajectory MSE.
    cross=defaultdict(lambda:defaultdict(float))
    ticks=read(R/'existing_ticks.csv')
    assert len(ticks)==144000
    for t in ticks:
        g=cross[(t['environment'],t['policy'])]
        harmful=float(t['excess_both'])>1e-12
        outside=int(t['outside_latest_two'])>0
        expired=int(t['expired_sources'])>0
        g['ticks']+=1
        if harmful: g[('outside' if outside else 'latest_two')+('_with_expiration' if expired else '_without_expiration')+'_ticks']+=1
    crossrows=[]
    for (e,p),v in cross.items():
        crossrows.append(dict(environment=e,policy=p,**{k:int(v[k]) for k in ['ticks','outside_with_expiration_ticks','outside_without_expiration_ticks','latest_two_with_expiration_ticks','latest_two_without_expiration_ticks']}))
    write('tcns_expiration_crosstab.csv', crossrows)
    selection=[]
    for r in core:
        if r['policy']=='COMPAT':
            selection.extend(dict(tag=r['tag'],**s) for s in read(BASE/'runs'/r['tag']/'selection.csv'))
    assert len(selection)==1920
    write('tcns_local_certificates.csv', selection)
    strict=sum(float(s['risk_host'])>float(s['risk_selected'])+1e-12 for s in selection)
    assert strict==22
    # Enqueue phase comparison: use exactly the matching four-instance/seed keys.
    sensitivities=[]
    for r in radio:
        if float(r['processing_s'])!=.001: continue
        h=next(x for x in core if all(x[k]==r[k] for k in ['environment','policy','observation_seed','network_seed']))
        sensitivities.append(dict(environment=r['environment'],policy=r['policy'],instance=r['observation_seed'],radio_seed=r['network_seed'],
                                  mse_1ms=float(r['mse']),mse_40ms=float(h['mse']),
                                  pssch_1ms=float(r['pssch_tx_prb_slots']),pssch_40ms=float(h['pssch_tx_prb_slots'])))
    assert len(sensitivities)==8
    write('tcns_processing_pairs.csv',sensitivities)
    # Fig. 1: supplied artwork and LaTeX overlay are packaged at the root.
    # Fig. 2: cycle, strict witness, hypergraph.
    fig,axs=plt.subplots(1,3,figsize=(7.16,2.13),constrained_layout=True)
    ax=axs[0];ax.set(xlim=(-1.5,1.5),ylim=(-1.45,1.6));ax.axis('off');ax.set_title('(a) Cycle boundaries')
    pts=[(-.8,.7),(.8,.7),(.8,-.7),(-.8,-.7)];bits=[0,0,1,1]
    for i,(x,y) in enumerate(pts):
        nx,ny=pts[(i+1)%4]
        ax.annotate('',xy=(nx*.80,ny*.80),xytext=(x*.80,y*.80),arrowprops={'arrowstyle':'->','color':'#b34c35' if bits[i]!=bits[(i+1)%4] else '#73808b','lw':1.8})
        ax.add_patch(Circle((x,y),.24,color='#326a97' if bits[i]==0 else '#da9446'))
        ax.text(x,y,str(bits[i]),color='white',ha='center',va='center')
    ax.text(0,0,r'$J-J^*=\Delta$',ha='center');ax.text(0,-1.30,'2 boundaries; 1 missing/duplicate pair',ha='center',fontsize=7)
    ax=axs[1];vals=[37/875,383/19825,73/1675,203/825]
    ax.bar(range(4),vals,color=['#326a97','#589572','#da9446','#da9446'])
    ax.set_xticks(range(4));ax.set_xticklabels(['old','new','mix 1','mix 2']);ax.set_ylabel('Posterior cost at B');ax.set_title('(b) Strict non-tie witness')
    ax.axhline(vals[0],color='#304154',ls='--',lw=.8);ax.set_ylim(0,.28)
    ax=axs[2];ax.set(xlim=(0,5),ylim=(0,3.5));ax.axis('off');ax.set_title('(c) Target interaction scopes')
    ax.add_patch(Ellipse((1.5,2.1),2.9,1.05,fc='#e5eef5',ec='#326a97'))
    ax.add_patch(Ellipse((4.1,2.1),1.25,1.05,fc='#f6e7d5',ec='#da9446'))
    for i,x in enumerate([.5,1.5,2.5,3.8,4.4]):ax.text(x,2.1,str(i+1),ha='center',va='center',bbox=dict(boxstyle='circle,pad=.23',fc='white',ec='#73808b'))
    ax.text(1.5,2.96,r'$H_1=\{1,2,3\}$',ha='center');ax.text(4.1,2.96,r'$H_2=\{4,5\}$',ha='center',fontsize=7)
    ax.text(2.5,.98,r'$2^3+2^2=12$ masks, versus $2^5=32$',ha='center',fontsize=7)
    ax.text(2.5,.35,r'$g_1=(1+z_1+z_2+z_3)^{-1}$: cubic term',ha='center',fontsize=7)
    save(fig,'tcns_fig2_structure')
    # Fig. 3 includes every archived policy/environment cell (40 runs per cell).
    fig,axs=plt.subplots(1,3,figsize=(7.16,4.1),sharey=True,constrained_layout=True)
    y=np.arange(18); short=dict(NO_FEEDBACK='N',COV_PERIODIC='P',COV_EVENT='E',GRANT='G',GOQ='Q',PRESCRIPTION='R')
    labels=[f"{dict(reliable='L',partial='P',contention='C')[g['environment']]}/{short[g['policy']]}" for g in groups]
    same=np.array([100*g['identical_updates']/g['updates'] if g['updates'] else 0 for g in groups])
    stricts=np.array([100*g['strict_improvements']/g['updates'] if g['updates'] else 0 for g in groups])
    rest=np.array([100 if g['updates'] else 0 for g in groups])-same-stricts
    axs[0].barh(y,same,color='#90afc8',hatch='///',edgecolor='#52616b',lw=.3,label='Identical renewal');axs[0].barh(y,stricts,left=same,color='#589572',hatch='xxx',edgecolor='#52616b',lw=.3,label='Strict improvement')
    axs[0].barh(y,rest,left=same+stricts,color='#d2d5d8',edgecolor='#52616b',lw=.3,label='Other change')
    for k in [0,6,12]:axs[0].text(2,k,'no updates',va='center',fontsize=6.5)
    axs[0].set(yticks=y,yticklabels=labels,xlabel='Broadcasts (%)',title='(a) Complete-plan changes',xlim=(0,102));axs[0].invert_yaxis()
    axs[0].legend(loc='lower left',bbox_to_anchor=(0,1.10),frameon=False,ncol=1,fontsize=6.5)
    outside=np.array([g['outside_latest_two_seconds'] for g in groups]);within=np.array([g['latest_two_seconds'] for g in groups])
    axs[1].plot(outside,y,'s',ms=4,color='#b34c35',label='Outside latest two');axs[1].plot(within,y,'o',ms=3,color='#326a97',label='Latest two')
    axs[1].set_xscale('symlog',linthresh=.0025);axs[1].set(xlim=(-.001,25),xlabel='Harmful duration (s/run)',title='(b) Realized excess cost')
    axs[1].set_xticks([0,.01,.1,1,10]);axs[1].set_xticklabels(['0','.01','.1','1','10']);axs[1].legend(loc='lower left',bbox_to_anchor=(0,1.10),frameon=False,fontsize=6.5)
    axs[2].barh(y,[g['expired_source_seconds'] for g in groups],color='#da9446')
    axs[2].set(xlabel='Expired/default (source-s/run)',title='(c) Expiration exposure',xlim=(0,62))
    for ax in axs:
        for line in [5.5,11.5]:ax.axhline(line,color='#ccd1d6',lw=.8)
        ax.grid(axis='x',alpha=.15)
    save(fig,'tcns_fig3_prevalence')
    # Fig. 4: local certificate and all 24 matched outcomes; seeded points are not replicates of process instances.
    fig,axs=plt.subplots(1,3,figsize=(7.16,2.45),constrained_layout=True)
    rh=np.array([float(s['risk_host']) for s in selection]);rs=np.array([float(s['risk_selected']) for s in selection])
    axs[0].scatter(rh,rs,s=12,alpha=.45,color='#326a97');limit=max(rh.max(),rs.max())*1.05
    axs[0].plot([0,limit],[0,limit],'--',color='#73808b',lw=.8)
    axs[0].set(xlabel='Host certificate',ylabel='Selected certificate',title='(a) Local risk: 1920 decisions',xlim=(-limit*.03,limit),ylim=(-limit*.03,limit))
    axs[0].text(.05,.9,'22 strict reductions',transform=axs[0].transAxes,fontsize=7)
    colors=['#326a97','#b34c35','#589572']
    for eidx,e in enumerate(ENVS):
        pp=[p for p in pairs if p['environment']==e];pp=sorted(pp,key=lambda x:(x['instance'],x['radio_seed']))
        xx=eidx+np.linspace(-.18,.18,8)
        axs[1].scatter(xx,[p['mse_delta']*1e4 for p in pp],s=18,color=colors[eidx],marker=['o','s','^'][eidx],zorder=3)
        axs[2].scatter(xx,[p['resource_delta']/1000 for p in pp],s=18,color=colors[eidx],marker=['o','s','^'][eidx],zorder=3)
    for ax in axs[1:]:
        ax.axhline(0,color='#73808b',lw=.8);ax.set_xticks(range(3));ax.set_xticklabels(['Light','Partial','Contention']);ax.grid(axis='y',alpha=.2)
    axs[1].set(title='(b) All 24 paired MSE effects',ylabel=r'Compat. - host MSE ($10^{-4}$)')
    axs[2].set(title='(c) All 24 resource effects',ylabel=r'Compat. - host ($10^3$ RB-slots)')
    save(fig,'tcns_fig4_compatibility')
    summary=dict(archived_runs=len(old),targeted_runs=len(radio),primary_runs=len(core),matched_pairs=len(pairs),
        exact_equal_pairs=20,local_decisions=len(selection),strict_risk_reductions=strict,
        positive_host_risk=int((rh>1e-12).sum()),max_nominal_gap=max(float(s['nominal_selected'])-float(s['nominal_best']) for s in selection),
        max_risk_violation=float((rs-rh).max()),broadcasts=sum(int(r['updates']) for r in old),ticks=len(ticks),
        harmful_latest_two_ticks=sum(int(v['latest_two_with_expiration_ticks']+v['latest_two_without_expiration_ticks']) for v in crossrows),
        harmful_outside_ticks=sum(int(v['outside_with_expiration_ticks']+v['outside_without_expiration_ticks']) for v in crossrows),
        total_pssch=sum(int(r['pssch_tx_prb_slots']) for r in radio))
    (OUT/'tcns_evidence_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    hashes=[dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(sources)]
    write('tcns_input_hashes.csv',hashes)
    print(json.dumps(summary,indent=2))
    print('Processing sensitivity means:',[(p,mean([r for r in sensitivities if r['policy']==p],'mse_1ms'),mean([r for r in sensitivities if r['policy']==p],'mse_40ms')) for p in POL])

if __name__=='__main__': main()
