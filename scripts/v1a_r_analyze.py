"""Task R steps R2-R3: count-aligned (A) and verification-cost (B) re-analysis.

Input is the compact output of scripts/v1a_r_extract.py, which must have passed
reproduction. Writes a new directory with reanalysis.json, report_tables_ZH.md
and two figures. Exploratory only: the V1a formal decision is unchanged.
"""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

import numpy as np

from xtbflow.v1.data import file_hash
from xtbflow.v1.reanalysis import (B0_UNITS,atom_summary,budget_prefix,count_prefix,included_points,kappa_star,mode_key,
                                   paired_difference,parent_mean,renormalized_auc,summarize,unit_prefix)

ENDPOINT_FIELD={'joint_best_reference':'joint','event_best':'event'}
STATEMENT=('探索性分析：V1a 正式集已揭盲一次，本目录任何结果都不是确认性证据；'
           'V1a 正式决策 GO_V1b 及其冻结配置、数据和分析函数均不变。')


def pair_name(pair):return f'{pair[0]}-{pair[1]}'


def load(extract):
    streams={}
    with gzip.open(extract/'streams.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            r['completion_units']=np.asarray(r['completion_units'],dtype=float)
            for k in ('joint','event'):r[k]=np.asarray(r[k],dtype=bool)
            r['opens']={k:np.asarray(v,dtype=bool) for k,v in r['opens'].items()}
            streams[r['arm'],r['parent_id'],r['training_seed']]=r
    return streams


def axis_length(stream,axis,n):
    if axis=='completed_candidates':return count_prefix(len(stream['completion_units']),n)
    if axis.startswith('verification_units_'):return unit_prefix(stream['opens'][axis[len('verification_units_'):]],n)
    raise ValueError(axis)


def analysis_a(streams,cfg,order,groups,seeds,atoms,labels):
    out={}
    axes=[cfg['count_axis_main'],*cfg['count_axis_sensitivity']]
    for axis in axes:
        out[axis]={}
        for endpoint,field in ENDPOINT_FIELD.items():
            cells={}
            for arm in cfg['arms']:
                for n in cfg['count_grid']:
                    c={}
                    for p in order:
                        for s in seeds:
                            st=streams[arm,p,s];m=axis_length(st,axis,n)
                            c[p,s]=None if m is None else float(st[field][:m].any())
                    cells[arm,n]=c
            rates={}
            for arm in cfg['arms']:
                rates[arm]={}
                for n in cfg['count_grid']:
                    mean,_,_,dropped=parent_mean(cells[arm,n],order,groups,seeds)
                    rates[arm][str(n)]=dict(rate=mean,parent_coverage=1-len(dropped)/len(order),
                        censored_cell_fraction=sum(v is None for v in cells[arm,n].values())/len(cells[arm,n]))
            pairs={}
            for pair in cfg['pairs']:
                pairs[pair_name(pair)]={}
                for n in cfg['count_grid']:
                    summary,coverage,dropped=paired_difference(cells[pair[0],n],cells[pair[1],n],order,groups,seeds)
                    gone=set(dropped)
                    pairs[pair_name(pair)][str(n)]=dict(summary=summary,parent_coverage=coverage,
                        coverage_insufficient=coverage<cfg['count_coverage_min_parents'],
                        censored_parent_atoms=atom_summary([atoms[p] for p in order if p in gone]),
                        retained_parent_atoms=atom_summary([atoms[p] for p in order if p not in gone]))
            out[axis][endpoint]=dict(arm_rates=rates,pairs=pairs)
            if axis==cfg['count_axis_main']:
                strata={}
                for pair in cfg['operational']['strata_pairs']:
                    for n in cfg['operational']['strata_n']:
                        for name in sorted(set(labels.values())):
                            keep=[(p,g) for p,g in zip(order,groups) if labels[p]==name]
                            summary,coverage,_=paired_difference(cells[pair[0],n],cells[pair[1],n],
                                [p for p,_ in keep],[g for _,g in keep],seeds)
                            strata.setdefault(pair_name(pair),{}).setdefault(str(n),{})[name]=dict(
                                summary=summary,parent_coverage=coverage,n_parents_in_stratum=len(keep))
                out[axis][endpoint]['rarity_strata']=strata
    classes={}
    for arm in cfg['arms']:
        classes[arm]={}
        for n in cfg['count_grid']:
            item={}
            for cls in cfg['operational']['outcome_classes']:
                c={}
                for p in order:
                    for s in seeds:
                        st=streams[arm,p,s];m=count_prefix(len(st['completion_units']),n)
                        c[p,s]=None if m is None else float(np.mean([v==cls for v in st['proxy_status'][:m]]))
                mean,_,_,dropped=parent_mean(c,order,groups,seeds)
                item[cls]=mean
            item['parent_coverage']=1-len(dropped)/len(order)
            classes[arm][str(n)]=item
    return out,classes


def analysis_b(streams,cfg,order,groups,seeds,mode):
    arms=cfg['arms'];ks=cfg['kappa_budget_points_k'];w=cfg['auc_weights'];limit=cfg['censoring_max_fraction']
    result={}
    for endpoint,field in ENDPOINT_FIELD.items():
        per_kappa={}
        for kappa in cfg['kappa_grid_units']:
            hits={}
            for arm in arms:
                for p in order:
                    for s in seeds:
                        st=streams[arm,p,s];row=[]
                        for k in ks:
                            m=budget_prefix(st['completion_units'],st['opens'][mode],kappa,k*(B0_UNITS+kappa),st['cap_units'])
                            row.append(None if m is None else float(st[field][:m].any()))
                        hits[arm,p,s]=row
            frac={arm:[float(np.mean([hits[arm,p,s][i] is None for p in order for s in seeds])) for i in range(len(ks))]
                  for arm in arms}

            def included(group):return included_points(frac,group,limit)

            def auc_cells(arm,inc):
                cells={}
                for p in order:
                    for s in seeds:
                        h=hits[arm,p,s]
                        cells[p,s]=None if any(h[i] is None for i in range(len(ks)) if inc[i]) else \
                            renormalized_auc([0. if v is None else v for v in h],inc,w)
                return cells
            five=included(arms);item=dict(censored_fraction=frac,included_five_arm=five,
                                          included_points_five_arm=[k for k,i in zip(ks,five) if i])
            if any(five):
                cells={arm:auc_cells(arm,five) for arm in arms};table={}
                for arm in arms:
                    mean,_,_,dropped=parent_mean(cells[arm],order,groups,seeds)
                    table[arm]=dict(auc=mean,parent_coverage=1-len(dropped)/len(order))
                ranking=sorted(arms,key=lambda a:-table[a]['auc'])
                summary,coverage,_=paired_difference(cells[ranking[0]],cells[ranking[1]],order,groups,seeds)
                ci=summary.get('ci_two95')
                item.update(arm_auc=table,ranking=ranking,frontier=dict(
                    top=ranking[0],second=ranking[1],difference=summary,parent_coverage=coverage,
                    interval_excludes_zero=None if ci is None else bool(ci[0]>0 or ci[1]<0)))
            pairs={}
            for pair in cfg['pairs']:
                inc=included(pair);entry=dict(included=inc,included_points=[k for k,i in zip(ks,inc) if i])
                if any(inc):
                    summary,coverage,_=paired_difference(auc_cells(pair[0],inc),auc_cells(pair[1],inc),
                                                         order,groups,seeds)
                    entry.update(summary=summary,parent_coverage=coverage)
                pairs[pair_name(pair)]=entry
            item['pairs']=pairs;per_kappa[str(kappa)]=item
        stars={}
        for pair in cfg['pairs']:
            path=[(kappa,per_kappa[str(kappa)]['pairs'][pair_name(pair)].get('summary')) for kappa in cfg['kappa_grid_units']]
            stars[pair_name(pair)]=kappa_star(path)
        result[endpoint]=dict(kappa=per_kappa,kappa_star=stars)
    return result


def gpu_seconds(parents,order):
    """Seconds of GPU time per network unit (one f forward), summarized over parents."""
    def stats(v):
        v=np.asarray(v,dtype=float);return dict(median=float(np.median(v)),min=float(v.min()),max=float(v.max()))
    return dict(
        f_call_dev=stats([parents[p]['f_seconds_per_call_dev'] for p in order]),
        per_molecule_dev=stats([parents[p]['f_seconds_per_call_dev']/parents[p]['dev_batch_size'] for p in order]),
        per_molecule_frozen=stats([parents[p]['f_seconds_per_call_frozen']/parents[p]['frozen_batch_size'] for p in order]),
        dev_batch_size=sorted({parents[p]['dev_batch_size'] for p in order}),
        frozen_batch_size=sorted({parents[p]['frozen_batch_size'] for p in order}))


def fmt(s,digits=3):
    if s is None or s.get('estimate') is None:return '—'
    if 'ci_two95' not in s:return f"{s['estimate']:+.{digits}f}（无区间）"
    lo,hi=s['ci_two95'];return f"{s['estimate']:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]"


def tables(res,cfg):
    a=res['analysis_a'];main=cfg['count_axis_main'];ns=[str(n) for n in cfg['count_grid']]
    lines=['# V1a-R 自动生成表格（探索性）','',STATEMENT,'']
    for axis in a:
        for endpoint,block in a[axis].items():
            lines+=[f'## 分析 A：{axis} × {endpoint}','','| 臂 | '+' | '.join(f'n={n}' for n in ns)+' |',
                    '|---|'+'---:|'*len(ns)]
            for arm,row in block['arm_rates'].items():
                lines.append(f'| {arm} | '+' | '.join('—' if row[n]['rate'] is None else
                    f"{row[n]['rate']:.3f}" + ('' if row[n]['parent_coverage']>=cfg['count_coverage_min_parents'] else '†')
                    for n in ns)+' |')
            lines+=['','| 对比 | '+' | '.join(f'n={n}' for n in ns)+' |','|---|'+'---|'*len(ns)]
            for pair,row in block['pairs'].items():
                lines.append(f'| {pair} | '+' | '.join(fmt(row[n]['summary'])+f" ({row[n]['parent_coverage']:.0%}"
                    +(' 覆盖不足' if row[n]['coverage_insufficient'] else '')+')' for n in ns)+' |')
            lines.append('')
    lines+=['† 母体覆盖率 < 80%。括号内为母体覆盖率。','']
    for mode,block in res['analysis_b'].items():
        for endpoint,b in block.items():
            lines+=[f'## 分析 B：{mode} × {endpoint}','','| κ | 入选预算点 | '+' | '.join(cfg['arms'])+' | 第一−第二 |',
                    '|---:|---|'+'---:|'*len(cfg['arms'])+'---|']
            for kappa,item in b['kappa'].items():
                if 'arm_auc' not in item:
                    lines.append(f'| {kappa} | 无 | '+' | '.join('—' for _ in cfg['arms'])+' | — |');continue
                fr=item['frontier']
                lines.append(f"| {kappa} | {','.join(map(str,item['included_points_five_arm']))} | "+
                    ' | '.join(f"{item['arm_auc'][arm]['auc']:.3f}" for arm in cfg['arms'])+
                    f" | {fr['top']}−{fr['second']} {fmt(fr['difference'])} |")
            lines+=['','| κ | '+' | '.join(pair_name(p) for p in cfg['pairs'])+' |','|---:|'+'---|'*len(cfg['pairs'])]
            for kappa,item in b['kappa'].items():
                lines.append(f'| {kappa} | '+' | '.join(fmt(item['pairs'][pair_name(p)].get('summary'))+
                    ' ('+','.join(map(str,item['pairs'][pair_name(p)]['included_points']))+')' for p in cfg['pairs'])+' |')
            lines+=['','κ\\*：']
            for pair,star in b['kappa_star'].items():
                if star['kappa_star'] is None:
                    lines.append(f"- {pair}：网格内不翻转（κ=0 符号 {star['sign_at_zero']:+.0f}；最后可估 κ={star['last_estimated_kappa']}）")
                else:
                    lines.append(f"- {pair}：κ\\*={star['kappa_star']}（前一格 κ={star['kappa_before']}：{fmt(star['before'])}；"
                                 f"κ\\*：{fmt(star['at_kappa_star'])}）")
            lines.append('')
    return '\n'.join(lines)+'\n'


def figures(res,cfg,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    surface='#fcfcfb';ink='#0b0b0b';muted='#52514e';grid='#e4e3df'
    series={'A0':'#2a78d6','A1':'#eb6834','A2':'#1baf7a','B0':'#eda100','B1':'#e87ba4'}
    markers={'A0':'o','A1':'s','A2':'^','B0':'D','B1':'v'}
    plt.rcParams.update({'font.size':10,'axes.edgecolor':muted,'axes.labelcolor':ink,'xtick.color':muted,
                         'ytick.color':muted,'figure.facecolor':surface,'axes.facecolor':surface})
    a=res['analysis_a'][cfg['count_axis_main']];ns=cfg['count_grid']
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
    for ax,(endpoint,title) in zip(axes,[('joint_best_reference','Joint endpoint (best reference hit)'),('event_best','Event endpoint (best event hit)')]):
        for arm in cfg['arms']:
            rows=a[endpoint]['arm_rates'][arm]
            x=[n for n in ns if rows[str(n)]['rate'] is not None];y=[rows[str(n)]['rate'] for n in x]
            low=[rows[str(n)]['parent_coverage']<cfg['count_coverage_min_parents'] for n in x]
            ax.plot(x,y,color=series[arm],lw=2,marker=markers[arm],ms=7,mec=surface,mew=1.5,label=arm)
            for xi,yi,l in zip(x,y,low):
                if l:ax.plot(xi,yi,marker=markers[arm],ms=11,mfc='none',mec=series[arm],mew=1)
        ax.set_xscale('log',base=2);ax.set_xticks(ns);ax.set_xticklabels([str(n) for n in ns])
        ax.set_xlabel('completed candidates n');ax.set_title(title,color=ink,fontsize=10,loc='left')
        ax.grid(True,color=grid,lw=.8);ax.set_axisbelow(True)
        for side in ('top','right'):ax.spines[side].set_visible(False)
    axes[0].set_ylabel('hit rate within first n candidates')
    axes[1].legend(frameon=False,loc='lower right',fontsize=9)
    fig.text(.01,.01,'Exploratory. Mean over parents of available seeds. Open ring: parent coverage < 80%.',color=muted,fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'hit_rate_vs_completed_candidates.png',dpi=150);plt.close(fig)

    main=mode_key(cfg['verification_units']['main']);b=res['analysis_b'][main]['joint_best_reference']['kappa']
    kappas=cfg['kappa_grid_units'];arms=cfg['arms']
    grid_values=np.full((len(arms),len(kappas)),np.nan)
    for j,kappa in enumerate(kappas):
        item=b[str(kappa)]
        if 'arm_auc' in item:
            for i,arm in enumerate(arms):grid_values[i,j]=item['arm_auc'][arm]['auc']
    cmap=LinearSegmentedColormap.from_list('blue',['#cde2fb','#86b6ef','#3987e5','#256abf','#104281','#0d366b'])
    cmap.set_bad('#f0efec')
    fig,ax=plt.subplots(figsize=(11,3.8))
    im=ax.imshow(np.ma.masked_invalid(grid_values),cmap=cmap,aspect='auto',vmin=0,vmax=max(.05,np.nanmax(grid_values)))
    for i in range(len(arms)):
        for j in range(len(kappas)):
            v=grid_values[i,j]
            if np.isnan(v):ax.text(j,i,'none',ha='center',va='center',color=muted,fontsize=8)
            else:ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,
                         color='#ffffff' if v>.45*np.nanmax(grid_values) else ink)
    points=[','.join(map(str,b[str(k)]['included_points_five_arm'])) or '-' for k in kappas]
    ax.set_xticks(range(len(kappas)));ax.set_xticklabels([f'{k}\nk={pt}' for k,pt in zip(kappas,points)],fontsize=7)
    ax.set_yticks(range(len(arms)));ax.set_yticklabels(arms)
    ax.set_xlabel('kappa (network units per verification unit); k = budget points used')
    ax.set_title(f'AUC_kappa, joint endpoint, units {main}, budget points shared by all five arms (exploratory)',color=ink,fontsize=10,loc='left')
    for side in ('top','right','left','bottom'):ax.spines[side].set_visible(False)
    fig.colorbar(im,ax=ax,fraction=.025,pad=.01)
    fig.tight_layout();fig.savefig(out/'auc_kappa_heatmap.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--extract',type=Path,required=True)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--gate',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(a.config.read_text());extract=json.loads((a.extract/'extract_manifest.json').read_text())
    if extract['status']!='EXTRACT_REPRODUCTION_PASS':raise ValueError('extraction did not pass reproduction')
    if extract['config_sha256']!=file_hash(a.config):raise ValueError('config differs from the extraction config')
    if file_hash(a.extract/'streams.jsonl.gz')!=extract['streams_gz_sha256']:raise ValueError('extract altered')
    gate=json.loads(a.gate.read_text())
    if file_hash(a.gate)!=extract['hash_checks']['final/gate.json']['actual']:raise ValueError('gate differs')
    streams=load(a.extract);parents=extract['parents'];labels=extract['rarity']['labels']
    order=sorted(parents);groups=[parents[p]['split_group'] for p in order];atoms={p:parents[p]['n_atoms'] for p in order}
    seeds=cfg['operational']['source']['training_seeds']
    res=dict(schema='xtbflow-v1a-r-reanalysis/1',exploratory=True,statement=STATEMENT,
             config_version=cfg['version'],config_sha256=extract['config_sha256'],config_commit=extract['config_commit'],
             code_commit=extract['code_commit'],extract_streams_gz_sha256=extract['streams_gz_sha256'],
             v1a_freeze_sha256=gate['freeze_sha256'],n_parents=len(order),n_formula_groups=len(set(groups)),
             reproduction_extract=dict(passed=True,max_abs_deviation=extract['reproduction']['max_abs_deviation'],
                                       side_file_cross_check=extract['reproduction']['side_file_cross_check']))
    res['analysis_a'],res['outcome_classes']=analysis_a(streams,cfg,order,groups,seeds,atoms,labels)
    modes=[mode_key(cfg['verification_units']['main'])]+[mode_key(s) for s in cfg['verification_units']['sensitivity']]
    res['analysis_b']={m:analysis_b(streams,cfg,order,groups,seeds,m) for m in modes}
    # Reproduction 2: kappa = 0 must give V1a's AUC exactly, for every mode.
    worst=0.
    for m in modes:
        for endpoint,key in (('joint_best_reference','auc'),('event_best','event_auc')):
            item=res['analysis_b'][m][endpoint]['kappa']['0']
            if item['included_points_five_arm']!=cfg['kappa_budget_points_k']:worst=float('inf')
            for arm in cfg['arms']:worst=max(worst,abs(item['arm_auc'][arm]['auc']-gate['curves'][arm][key]))
        s=res['analysis_b'][m]['joint_best_reference']['kappa']['0']['pairs']['B1-A2']['summary']
        worst=max(worst,abs(s['estimate']-gate['primary']['estimate']),
                  *[abs(x-y) for x,y in zip(s['ci_two95'],gate['primary']['ci_two95'])])
    res['reproduction_kappa0']=dict(max_abs_deviation=worst,tolerance=cfg['operational']['reproduction_tolerance'],
                                    passed=worst<=cfg['operational']['reproduction_tolerance'])
    res['gpu_seconds_per_unit']=gpu_seconds(parents,order)
    (a.out/'reanalysis.json').write_text(json.dumps(res,indent=1,sort_keys=True,allow_nan=False)+'\n')
    (a.out/'report_tables_ZH.md').write_text(tables(res,cfg),encoding='utf-8')
    figures(res,cfg,a.out)
    print(json.dumps(dict(reproduction_kappa0=res['reproduction_kappa0'],
                          bytes=(a.out/'reanalysis.json').stat().st_size),indent=2))
    if not res['reproduction_kappa0']['passed']:sys.exit(3)


if __name__=='__main__':main()
