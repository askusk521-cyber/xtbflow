"""Plot the frozen per-k recall table without altering numerical analysis."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r=json.loads(a.results.read_text());fig,ax=plt.subplots(figsize=(9,6))
    for name,curve in sorted(r['rates'].items()):
        points=sorted((int(k),v) for k,v in curve.items() if v is not None)
        ax.plot([k for k,v in points],[v for k,v in points],marker='o',label=name,linestyle='-' if name.startswith('S_') else '--')
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32,64]);ax.set_xticklabels([1,2,4,8,16,32,64])
    ax.set(xlabel='Distinct events sent for verification (k)',ylabel='Best catalogue event recall',ylim=(0,1),title='Exploratory: enumeration vs frozen generation')
    ax.text(.01,.01,'Generator curves condition on uncensored seeds; high-k points may be descriptive only.',transform=ax.transAxes,fontsize=8)
    ax.grid(alpha=.2);ax.legend(ncol=2);fig.tight_layout();fig.savefig(a.out,dpi=160)


if __name__=='__main__':main()
