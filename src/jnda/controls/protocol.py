"""Sample ordering and the fixed development-only training budget rule."""
import hashlib
import math

ORDER_NAMESPACE='S8R_Ex01_cleanroom_joint_native_replication_v1_1'

DEV_N=1206

class ContractError(RuntimeError):pass

def need(value,code):
 if not value:raise ContractError(code)

def finite(value):return type(value) in (int,float) and math.isfinite(value)

def sha256(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def stable_order(sample_ids,seed,epoch):
 """Exact old ordering formula with an explicit formal seed, no2701 default."""
 need(seed in range(1701,1706) and type(epoch) is int and epoch>=0,'explicit_formal_order_seed_epoch')
 need(len(sample_ids)==len(set(sample_ids)),'duplicate_sample_id')
 return sorted(sample_ids,key=lambda sid:hashlib.sha256(f'{ORDER_NAMESPACE}\0{seed}\0{epoch}\0{sid}'.encode()).hexdigest())

def t3_group_extension(curves):
 """Consume complete dev metadata only; never select a LR or read test data."""
 need(set(curves)=={1701,1702,1703},'all_three_formal_T3_seeds_required')
 deltas={}
 for seed,rows in curves.items():
  by={r['epoch']:r for r in rows};need(len(rows)==len(by)==10 and set(by)==set(range(1,11)),'complete_T3_epochs1to10')
  for row in rows:
   need(row.get('seed')==seed and row.get('split')=='development' and row.get('full_dev') is True and
        row.get('unique_sample_count')==DEV_N and finite(row.get('pooled_miou')) and 0<=row['pooled_miou']<=1 and
        sha256(row.get('checkpoint_sha256')),'T3_full_unrounded_dev_identity')
  deltas[str(seed)]=max(by[e]['pooled_miou'] for e in range(6,11))-max(by[e]['pooled_miou'] for e in range(1,6))
 mean=math.fsum(deltas.values())/3;extend=mean>0.002
 return {'per_seed_best6to10_minus_best1to5':deltas,'equal_weight_mean':mean,'threshold':0.002,'strict_greater_than':True,
         'target_epochs_for_all_three':30 if extend else 10,'applies_to_seeds':[1701,1702,1703],
         'budget_or_scheduler_not_authorized_by_this_calculation':True,'test_feedback_used':False}

def selected_dev_checkpoint(rows):
 need(bool(rows),'complete_dev_curve_required')
 for row in rows:
  need(row.get('split')=='development' and row.get('full_dev') is True and row.get('unique_sample_count')==DEV_N and
       type(row.get('epoch')) is int and finite(row.get('pooled_miou')) and sha256(row.get('checkpoint_sha256')),'selected_checkpoint_requires_full_dev')
 need(len({r['epoch'] for r in rows})==len(rows),'duplicate_dev_epoch')
 # No rounding, and exact ties choose the earlier epoch.
 return min(rows,key=lambda row:(-row['pooled_miou'],row['epoch']))
