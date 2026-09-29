"""Rerun unchanged archived unit checks, redirecting only their result file."""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
p=root/'publication_track/installation_focus/test_mechanism.py'
s=p.read_text(encoding='utf-8')
old="(Path(__file__).parent/'results/tests.json')"
assert s.count(old)==1
s=s.replace(old,"(Path(__file__).resolve().parents[2]/'results/tcns_method_checks.json')")
exec(compile(s,str(p),'exec'),{'__name__':'__main__','__package__':'publication_track.installation_focus','__file__':str(p)})
