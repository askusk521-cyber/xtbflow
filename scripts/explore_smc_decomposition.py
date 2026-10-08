"""Posthoc availability/ranking-loss identity; no new scientific scoring."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from xtbflow.v1.metrics import cluster_summary


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def git(*args):return subprocess.check_output(['git',*args],text=True).strip()
def dump(p,v):Path(p).write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,default=Path('configs/explore/smc_decomposition.json'));parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not git('status','--porcelain') and git('branch','-r','--contains','HEAD')
    cfg=load(a.config);assert sha(cfg['input'])==cfg['input_sha256'];r=load(cfg['input'])
    parentpath='/home/lhshen/xtbflow-runs/v1a-20261007/data/parent_catalog.json'
    assert sha(parentpath)==r['provenance']['inputs'][parentpath]
    parents={p['parent_id']:p for p in load(parentpath)};pm=r['parent_metrics'];arms=sorted(pm['raw']);pids=sorted(pm['raw']['none']['H'])
    assert len(pids)==62 and len(arms)==13
    def summary(v):return cluster_summary([v[p] for p in pids],[parents[p]['split_group'] for p in pids])
    values={};absolute={};comparison={};error=0.
    for ranker,table in pm.items():
        values[ranker]={};absolute[ranker]={};comparison[ranker]={}
        for arm in arms:
            h=table[arm]['H'];availability=table[arm]['hit_infinity']
            assert set(h)==set(availability)==set(pids)
            assert all(-1e-12<=h[p]<=availability[p]+1e-12<=1+1e-12 for p in pids)
            assert availability==pm['raw'][arm]['hit_infinity']
            loss={p:availability[p]-h[p] for p in pids}
            vals=dict(H=h,availability=availability,ranking_loss=loss)
            values[ranker][arm]=vals;absolute[ranker][arm]={k:summary(v) for k,v in vals.items()}
        for step in (.6,.7,.8,.9):
            x,y,z=f'relaxed_{step:.1f}',f'raw_{step:.1f}',f'random_{step:.1f}'
            for left,right in ((x,'none'),(y,'none'),(x,z),(y,z),(x,y),(z,'none')):
                delta={m:{p:values[ranker][left][m][p]-values[ranker][right][m][p] for p in pids} for m in ('H','availability','ranking_loss')}
                error=max(error,max(abs(delta['H'][p]-delta['availability'][p]+delta['ranking_loss'][p]) for p in pids))
                comparison[ranker][left+'-'+right]={m:summary(v) for m,v in delta.items()}
    assert error<=1e-12
    result=dict(version=cfg['version'],exploratory=True,absolute=absolute,comparisons=comparison,max_identity_error=error,parent_values=values,
        interpretation=cfg['interpretation'],provenance=dict(source_commit=git('rev-parse','HEAD'),dirty=False,slurm_job_id=None,workers=1,
        config_sha256=sha(a.config),inputs={cfg['input']:sha(cfg['input']),parentpath:sha(parentpath)}))
    a.out.mkdir(parents=True,exist_ok=False);dump(a.out/'results.json',result)
    def fmt(s):return f"{s['estimate']:+.6f} [{s['ci_two95'][0]:+.6f}, {s['ci_two95'][1]:+.6f}]"
    lines=['本报告全部结果为探索性。以下是固定候选池的事后代数分解，不是新增生成实验或因果识别。','',
        '# 最佳通道可用性与排序损失','',
        '定义：可用性=hit@∞，排序损失=hit@∞−H，因此ΔH=Δ可用性−Δ排序损失。每个母体先平均三个训练seed，再计算配对差及分子式聚类95%区间。',
        '可用性是候选池内完美排序的上限；它不是可实施的oracle筛选，也不认证化学过渡态。分量区间不是独立的，不能用区间端点直接相减。','',
        '| 终点排序 | 比较 | ΔH | Δ可用性 | Δ排序损失 |','|---|---|---|---|---|']
    for ranker,cc in sorted(comparison.items()):
        for name,mm in sorted(cc.items()):lines.append('| '+ranker+' | '+name+' | '+' | '.join(fmt(mm[m]) for m in ('H','availability','ranking_loss'))+' |')
    lines+=['','全部13臂、两个排序器与24组比较均报告。原数据只读，输入哈希校验通过；逐母体分解恒等式最大误差见results.json。结果只描述损失来自哪个代数分量，不能推断某个组件的因果效应。']
    (a.out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    print('decomposition completed; all identity and coverage gates passed')


if __name__=='__main__':main()
