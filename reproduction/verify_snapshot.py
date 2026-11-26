"""Verify preserved source bytes and print the selected tested configuration; no inference."""
from pathlib import Path
import hashlib,json,ast
root=Path(__file__).resolve().parents[1]
m=json.loads((root/'reproduction/source-manifest.json').read_text(encoding='utf-8'))
count=0
for name,entry in m['files'].items():
 p=root/name
 assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
 if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'),filename=name);count+=1
v=json.loads((root/'reproduction/ACTIVE_VERSION.json').read_text(encoding='utf-8'))
print(json.dumps({'verified_files':len(m['files']),'python_files_parsed':count,'version':v},ensure_ascii=False,indent=2))
