"""Prepare frozen decision suites without placing targets in model messages."""
import argparse
import hashlib
import json
from pathlib import Path
from decision_schema import compile_row, _options, _answer_value

SUITES = [('easy','jevbench/easy.jsonl',48,48), ('original','jevbench/original.jsonl',72,72),
          ('hard','jevbench/hard.jsonl',111,111), ('agnews','agnews/test.jsonl',7600,7600),
          ('toolace','toolace/test.jsonl',310,310), ('typed','typed_decisions/test.jsonl',400,2000),
          ('wildjailbreak','wildjailbreak/test.jsonl',2210,2210)]


def pack(row):
    assert not row.get('images'), 'This acceptance recipe is text-only'
    clean = {'state': row['state'], 'questions': {
        name: {k:v for k,v in q.items() if k not in ['answer','target','gold']}
        for name,q in row['questions'].items()}}
    compiled = compile_row(clean, include_targets=False)
    choices = {f:[k for k,_ in _options(clean['questions'][f])] for f in compiled.fields}
    gold = {f:_answer_value(row['questions'][f], row.get('targets',{}).get(f)) for f in compiled.fields}
    assert all(gold[f] in choices[f] for f in compiled.fields)
    return {'case_id':str(row['id']), 'messages':compiled.messages, 'fields':list(compiled.fields),
            'candidates':{f:list(compiled.symbols[f]) for f in compiled.fields},
            'targets':{f:compiled.symbols[f][choices[f].index(gold[f])] for f in compiled.fields}}


def normalize(q):
    assert set(q) <= {'type','instructions','criteria'}
    q=dict(q)
    if not isinstance(q['instructions'],str):q['instructions']=json.dumps(q['instructions'])
    if q['type']=='choice' and isinstance(q['criteria'],list):q['criteria']=dict.fromkeys(q['criteria'],'')
    def render(v):return '' if v is None else v if isinstance(v,str) else json.dumps(v,ensure_ascii=False)
    c=q.get('criteria')
    if isinstance(c,dict):q['criteria']={str(k):render(v) for k,v in c.items()}
    elif isinstance(c,list):q['criteria']=[render(v) for v in c]
    return q


def main():
    p=argparse.ArgumentParser();p.add_argument('--official-root',type=Path,required=True)
    p.add_argument('--broad',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(exist_ok=False);manifest={'official':[],'broad':[],'source_sha256':{},'excluded':[]}
    def save(name,rows,decisions,kind,batch):
        keys=[(r['case_id'],f) for r in rows for f in r['fields']]
        assert len(keys)==len(set(keys))==decisions
        path=a.output/(name+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        manifest[kind].append({'name':name,'file':path.name,'expected_rows':len(rows),'expected_decisions':decisions,'batch_size':batch,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    for name,file,count,decisions in SUITES:
        path=a.official_root/file;manifest['source_sha256'][file]=hashlib.sha256(path.read_bytes()).hexdigest()
        raw=[json.loads(line) for line in path.read_text().splitlines()];assert len(raw)==count
        rows=[]
        for line_index,row in enumerate(raw):
            if 'questions' not in row:
                row={'id':row['id'],'state':row['state'],'questions':{'decision':row['question']},'targets':{'decision':{'label':row['expected']}}}
            record=pack(row)
            record["source_case_id"]=record["case_id"]
            record["case_id"]=name+":"+str(line_index)
            rows.append(record)
        save('official-'+name,rows,decisions,'official',8)
    manifest['source_sha256']['broad']=hashlib.sha256(a.broad.read_bytes()).hexdigest()
    for name,suite in sorted(json.loads(a.broad.read_text())['suites'].items()):
        rows=[]
        for i,case in enumerate(suite['cases']):
            for f,q in case['questions'].items():
                q=normalize(q);options=_options(q);gold=case['gold'][f];assert 0<=gold['idx']<len(options)
                if 'keys' in gold:assert list(map(str,gold['keys']))==(['false','true'] if q['type']=='noul' else [k for k,_ in options])
                if len(options)>62:
                    manifest['excluded'].append({'suite':name,'case_index':i,'field':f,'reason':'candidate_limit_62'});continue
                rows.append(pack({'id':name+':'+str(i)+':'+f,'state':case['state'],'questions':{f:q},'targets':{f:{'label':options[gold['idx']][0]}}}))
        if rows:save('broad-'+name,rows,len(rows),'broad',1)
    assert len(manifest['official'])==7 and sum(x['expected_decisions'] for x in manifest['official'])==12351
    assert len(manifest['broad'])==49 and sum(x['expected_decisions'] for x in manifest['broad'])==17416 and len(manifest['excluded'])==500
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
