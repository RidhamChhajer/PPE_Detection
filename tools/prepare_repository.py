"""Read-only Git audit and local source/release packaging; never push or commit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_DIRS = ('app','configs','tools','tests','MD DOC','.github')
SUFFIXES = {'.py','.html','.css','.js','.json','.md','.yml','.yaml','.ps1'}


def source_files():
    files = [ROOT/'README.md',ROOT/'.gitignore',*ROOT.glob('requirements-*.txt')]
    for directory in ALLOWED_DIRS:
        files += [p for p in (ROOT/directory).rglob('*') if p.is_file() and p.suffix in SUFFIXES and '__pycache__' not in p.parts]
    return sorted(set(files))


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT).decode('utf-8',errors='replace')


def findings(content,path):
    records = []
    for number,line in enumerate(content.splitlines(),1):
        if re.search(r'(?<![A-Za-z])[A-Za-z]:[\\/]|/Users/|/home/[A-Za-z]',line):
            records.append(dict(path=path,line=number,kind='absolute_path_review'))
        if re.search(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|sk-[A-Za-z0-9]{32,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----)',line):
            records.append(dict(path=path,line=number,kind='potential_secret'))
    return records


def audit(files):
    history = []
    risky_objects = []
    objects = git('rev-list','--objects','--all').splitlines()
    seen = set()
    for record in objects:
        parts = record.split(' ',1)
        if len(parts)!=2: continue
        oid,path = parts
        if oid in seen: continue
        seen.add(oid)
        if git('cat-file','-t',oid).strip() != 'blob': continue
        size = int(git('cat-file','-s',oid))
        if size > 10*1024*1024 or Path(path).suffix.lower() in ('.pth','.onnx','.engine','.mp4','.jpg','.png') or path.startswith(('data/','artifacts/','.venv/','work_dirs/')):
            risky_objects.append(dict(path=path,bytes=size,object=oid))
        if size < 2*1024*1024:
            history.extend(dict(object=oid,**f) for f in findings(git('cat-file','blob',oid),path))
    current = []
    for path in files: current.extend(findings(path.read_text(encoding='utf-8'),path.relative_to(ROOT).as_posix()))
    return dict(status=git('status','--short'),tracked_files=git('ls-files').splitlines(),
                ignored_roots=git('status','--ignored','--short').splitlines(),
                historical_risky_objects=risky_objects,historical_findings=history,current_findings=current)


def check_links(files):
    bad = []
    for path in files:
        if path.suffix != '.md': continue
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            target=unquote(target.strip('<>').split('#')[0])
            if not target or '://' in target: continue
            if not (path.parent/target).exists(): bad.append(f'{path.name}: {target}')
    if bad: raise ValueError('Broken documentation links: '+repr(bad))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path)
    parser.add_argument('--release-dir',type=Path,default=ROOT/'release_assets/v1')
    args=parser.parse_args()
    files=source_files()
    check_links(files)
    result=audit(files)
    (ROOT/'local').mkdir(exist_ok=True)
    (ROOT/'local/publication_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    args.release_dir.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.release_dir/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,path.relative_to(ROOT))
    if args.snapshot:
        if args.snapshot.exists(): raise ValueError('Use a new clean source snapshot path')
        for path in files:
            target=args.snapshot/path.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
    assets=sorted(p for p in args.release_dir.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
    sums=[]
    for path in assets:
        with path.open('rb') as stream:
            digest=hashlib.file_digest(stream,'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(stream.read()).hexdigest()
        sums.append(digest+'  '+path.name)
    (args.release_dir/'SHA256SUMS.txt').write_text('\n'.join(sums)+'\n',encoding='utf-8')
    inventory=dict(source_files=[p.relative_to(ROOT).as_posix() for p in files],release_assets=[dict(filename=p.name,bytes=p.stat().st_size) for p in assets])
    (ROOT/'local/release_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
    print(f'Source files: {len(files)}; release assets: {len(assets)}; links PASS')
    print(f'History binary/private/large candidates: {len(result["historical_risky_objects"])}; potential secrets: {sum(f["kind"]=="potential_secret" for f in result["historical_findings"])}; historical path references: {sum(f["kind"]=="absolute_path_review" for f in result["historical_findings"])}')
    print('Detailed path-only audit: local/publication_audit.json. No Git state or history modified.')


if __name__=='__main__': main()
