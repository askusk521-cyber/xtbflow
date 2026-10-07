"""Logical network-call ledger independent of physical GPU batching."""
from dataclasses import dataclass,field
import math


BUDGETS=(4,8,16,32,64)


@dataclass
class Ledger:
    weights: dict
    cap: float=3200.
    spent: float=0.
    operations: list=field(default_factory=list)
    attempts: list=field(default_factory=list)

    def charge(self,attempt,operation,count=1):
        weight=float(self.weights[operation]);cost=weight*count
        if count<0 or not math.isfinite(cost) or cost<0:
            raise ValueError('invalid network cost')
        if self.spent+cost>self.cap+1e-8:return False
        self.spent+=cost
        self.operations.append(dict(attempt_id=attempt,operation=operation,count=count,
                                    cost_units=cost,cumulative_units=self.spent))
        return True

    def can_charge_step(self,role,guidance=False,score_kind='X'):
        cost=self.weights[role]
        if guidance:cost+=3*(self.weights[score_kind]+self.weights[score_kind+'_back'])
        return self.spent+cost<=self.cap+1e-8

    def step(self,attempt,role,guidance=False,score_kind='X'):
        if not self.can_charge_step(role,guidance,score_kind):return False
        assert self.charge(attempt,role)
        if guidance:
            assert self.charge(attempt,score_kind,3)
            assert self.charge(attempt,score_kind+'_back',3)
        return True

    def complete(self,attempt,candidate_id,status):
        row=dict(attempt_id=attempt,candidate_id=candidate_id,status=status,
                 completion_units=self.spent)
        self.attempts.append(row)
        return row


def prefixes(candidates,budgets=BUDGETS):
    return {str(k):[c for c in candidates if c['completion_units']<=50*k+1e-8] for k in budgets}


def weights_for_atoms(table,n):
    for row in table['bins']:
        if row['min_atoms']<=n<=row['max_atoms']:
            return row['weights']
    raise ValueError(f'no frozen cost bin for {n} atoms')
