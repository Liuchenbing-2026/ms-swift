"""Run a small-model execution check, then the fixed completed 35B checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from tune_microbatch import check_idle_resume, checkpoint_files


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);plan=json.loads(Path(p.parse_args().plan).read_text())
    root=Path(plan['host_output']);root.mkdir(exist_ok=False)
    state={'status':'checking','test_used_for_selection':False}
    def persist():
        state['observed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
        temp=root/'state.tmp';temp.write_text(json.dumps(state,indent=2));temp.replace(root/'state.json')
    def run(name,model,resume,config):
        check_idle_resume(plan)
        for path,digest in plan['pinned_files'].items():
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,'Pinned file changed'
        env=dict(plan['environment']);env.update(MODEL_PATH=model,RESUME_FROM=resume,
            DECISION_EVAL_ONLY='1',DECISION_EVAL_CONFIG=config,MICROBATCH='4',
            OUTPUT_DIR=plan['container_output']+'/'+name,DECISION_PROBE_UPDATES='0',SAVE_STRATEGY='no',EVAL_STRATEGY='no')
        cmd=['docker','exec']
        for k,v in env.items():cmd+=['-e',k+'='+str(v)]
        cmd += [plan['container'],'timeout','--signal=TERM','--kill-after=60s',str(3600 if name=='smoke' else 43200),'bash',plan['container_launcher']]
        state.update(status='running',stage=name,command=cmd);persist()
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            state['child_pid']=proc.pid;persist();rc=proc.wait()
        assert rc==0,name+' failed; preserve log and weights'
        plan['previous_outputs'].append(env['OUTPUT_DIR']);check_idle_resume(plan)
        report=json.loads((root/name/'evaluation.json').read_text())
        assert report['status']=='complete' and report['optimizer_updates_performed']==0
        assert report['checkpoint_step']==(0 if name=='smoke' else 600)
        return report
    persist()
    try:
        before=checkpoint_files(plan['host_checkpoint'],600)
        smoke=run('smoke',plan['smoke_model'],'',plan['smoke_config'])
        state['smoke_complete']=True;persist()
        result=run('full',plan['model'],plan['container_checkpoint'],plan['full_config'])
        assert result['scope']=='official7_and_broad49'
        official={k:v for k,v in result['suites'].items() if k.startswith('official-')}
        broad={k:v for k,v in result['suites'].items() if k.startswith('broad-')}
        assert len(official)==7 and sum(v['rows'] for v in official.values())==10751
        assert sum(v['total'] for v in official.values())==12351
        assert len(broad)==49 and sum(v['total'] for v in broad.values())==17416
        assert before==checkpoint_files(plan['host_checkpoint'],600),'Read-only checkpoint changed'
        state.update(status='complete',limitation='Accuracy-only BF16 FSDP assessment; no training or test-guided selection')
    except Exception as error:
        state.update(status='failed',error=str(error));raise
    finally:persist()

if __name__=='__main__':main()
