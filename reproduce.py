"""Standalone numerical reproduction; no simulator or project checkout required."""
from pathlib import Path
import csv,json,hashlib,subprocess,sys,math,statistics
ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (ROOT/'results'/name).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def avg(rows,key):return statistics.mean(float(r[key]) for r in rows)
for row in read(ROOT/'manifest_sha256.csv'):
 assert hashlib.sha256((ROOT/row['path']).read_bytes()).hexdigest()==row['sha256'],row['path']
for d in ['results','figures']:(ROOT/d).mkdir(exist_ok=True)
for script in ['tcns_structural_checks.py','tcns_method_checks.py','tcns_build_evidence.py']:
 subprocess.run([sys.executable,'-B',str(ROOT/'scripts'/script)],cwd=ROOT,check=True)
comparisons=0
for ref in (ROOT/'expected').glob('*.csv'):
 actual=read(ROOT/'results'/ref.name);expected=read(ref)
 assert len(actual)==len(expected),ref.name
 for a,b in zip(actual,expected):
  assert a.keys()==b.keys(),ref.name
  for k in a:
   try:ok=math.isclose(float(a[k]),float(b[k]),rel_tol=1e-11,abs_tol=1e-12)
   except ValueError:ok=a[k]==b[k]
   assert ok,(ref.name,k,a[k],b[k])
   comparisons+=1
base=ROOT/'publication_track/installation_focus/results'
radio=read(base/'radio_results.csv')
core=[r for r in radio if float(r['processing_s'])==.04 and r['policy']!='PHASED']
timing=read(base/'decision_timing.csv')
print('Timing input columns:',list(timing[0]))
summary=json.loads((base/'summary.json').read_text())
timingrows=[]
for env in ['reliable','partial','contention']:
 for policy in ['GRANT_KEEP','COV_EVENT','OLD_RENEWAL','COMPAT']:
  rr=[r for r in core if r['environment']==env and r['policy']==policy]
  # Per-run quantiles are frozen timing observations, not new hardware timing.
  key='runtime_p95_s'
  value=statistics.median(float(r[key]) for r in rr)
  if '_s' in key and '_ms' not in key:value*=1000
  assert math.isclose(value,summary['metrics'][env+'|'+policy]['median_run_p95_ms'],abs_tol=1e-8)
  timingrows.append(dict(environment=env,policy=policy,median_run_p95_ms=value))
write('table_III_timing.csv',timingrows)
stage=[r for r in radio if r['policy']=='PHASED']
write('staged_means.csv',[dict(runs=len(stage),mse=avg(stage,'mse'),pssch_rb_slots=avg(stage,'pssch_tx_prb_slots'))])
jensen=read(base/'existing_jensen.csv')
full=read(base/'full_book_jensen.csv')
assert len(jensen)==782 and sum(int(r['choice_diff']) for r in jensen)==33
assert math.isclose(max(float(r['max_gap']) for r in jensen),0.04537671967668617,abs_tol=1e-14)
for service,changed,gap in [(0.5,480,0.10952553672496956),(0.75,523,0.09848430488088349)]:
 rows=[r for r in full if float(r['service'])==service]
 assert len(rows)==782 and sum(int(r['choice_diff']) for r in rows)==changed
 assert math.isclose(max(float(r['max_gap']) for r in rows),gap,abs_tol=1e-14)
protocol=[]
for row in stage:protocol.extend(read(ROOT/'publication_track/installation_focus/runs'/row['tag']/'protocol_events.csv'))
from collections import Counter
eventkey=next(k for k in protocol[0] if k in ['event','kind'])
events=dict(Counter(r[eventkey] for r in protocol))
assert events==summary['protocol_events'],events
write('jensen_diagnostics.csv',[dict(contexts=782,changed=33,max_gap=max(float(r['max_gap']) for r in jensen))])
report=dict(status='passed',csv_cells_compared=comparisons,
 input_radio_runs=len(radio),archived_runs=len(read(base/'existing_summary.csv')),
 jensen_contexts=len(jensen),full_book_rows=len(full),
 scope='Frozen-data reanalysis and mathematical/protocol checks; no new radio simulation')
evidence=json.loads((ROOT/'results/tcns_evidence_checks.json').read_text())
expected=json.loads((ROOT/'expected/tcns_evidence_checks.json').read_text())
assert evidence==expected
assert evidence['max_nominal_gap']==0.00048529018310453087
(ROOT/'results/reproduction_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
