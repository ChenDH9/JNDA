"""Sufficient-count reduction for class, boundary and weed-patch analyses."""
import math

import numpy as np

BOUND_KEYS=[(c,r) for c in (1,2) for r in (1,2,4)]

BOUND_FIELDS=['pred_boundary_total','gt_boundary_total','pred_matched','gt_matched','excluded_pred_contours','excluded_gt_contours']

PATCH_FIELDS=['patches','pixels','covered_pixels','patches_at_least_half','images_with_patches','coverage_sum']

METRICS=[f'class{c}_{m}' for c in range(3) for m in ['IoU','precision','recall','F1']]

METRICS += [f'class{c}_boundary_{m}_r{r}' for c,r in BOUND_KEYS for m in ['precision','recall','F1']]

METRICS += [f'{g}_weed_{m}' for g in ['small','tiny'] for m in ['pixel_recall','mean_patch_coverage','fraction_patches_at_least_half']]

assert len(METRICS)==36

FEATURES=[f'cm_{a}_{b}' for a in range(3) for b in range(3)]

FEATURES += [f'b{c}_r{r}_{k}' for c,r in BOUND_KEYS for k in BOUND_FIELDS]

FEATURES += [f'{g}_{k}' for g in ['small','tiny'] for k in PATCH_FIELDS]

assert len(FEATURES)==57

NONINTEGER_COLUMNS={50,56}

def feature_row(record):
    v=[int(x) for row in record['confusion_matrix'] for x in row]
    for c,r in BOUND_KEYS:
        b=next(x for x in record['boundary_counts'] if x['class_id']==c and x['radius_pixels']==r)
        v.extend(int(b[k]) for k in BOUND_FIELDS)
    patches=record['weed_patches']['records']
    for top in [256,16]:
        group=[x for x in patches if x['area_pixels']<=top]
        v.extend([len(group),sum(x['area_pixels'] for x in group),sum(x['covered_pixels'] for x in group),
                  sum(x['coverage_at_least_half'] for x in group),int(bool(group)),math.fsum(x['coverage'] for x in group)])
    assert len(v)==57
    return np.asarray(v,dtype=np.float64)

def safe_ratio(a,b):
    a,b=np.asarray(a,dtype=np.float64),np.asarray(b,dtype=np.float64)
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=b!=0)

def metrics_from_features(features):
    a=np.asarray(features,dtype=np.float64)
    assert a.ndim==2 and a.shape[1]==57
    cm=a[:,:9].reshape(-1,3,3);support=cm.sum(axis=2);predicted=cm.sum(axis=1)
    values=[]
    for c in range(3):
        tp=cm[:,c,c];gt=support[:,c];pr=predicted[:,c]
        values.extend([safe_ratio(tp,gt+pr-tp),safe_ratio(tp,pr),safe_ratio(tp,gt),safe_ratio(2*tp,gt+pr)])
    for k in range(6):
        pt,gt,pm,gm=a[:,9+6*k:13+6*k].T
        precision=safe_ratio(pm,pt);recall=safe_ratio(gm,gt)
        f1=np.zeros(len(a));both=(pt>0)&(gt>0);den=precision+recall
        use=both&(den>0);f1[use]=2*precision[use]*recall[use]/den[use]
        f1[(pt==0)&(gt==0)]=np.nan
        values.extend([precision,recall,f1])
    for k in range(2):
        n,area,hits,half,images,cov=a[:,45+6*k:51+6*k].T
        values.extend([safe_ratio(hits,area),safe_ratio(cov,n),safe_ratio(half,n)])
    return np.stack(values,axis=1)

def metric_denominators(features):
    a=np.asarray(features,dtype=np.float64);cm=a[:9].reshape(3,3)
    gt=cm.sum(axis=1);pr=cm.sum(axis=0);out={}
    for c in range(3):
        den=[gt[c]+pr[c]-cm[c,c],pr[c],gt[c],gt[c]+pr[c]]
        for m,d in zip(['IoU','precision','recall','F1'],den):out[f'class{c}_{m}']=int(d)
    for k,(c,r) in enumerate(BOUND_KEYS):
        pt,bt=a[9+6*k:11+6*k]
        out[f'class{c}_boundary_precision_r{r}']=int(pt)
        out[f'class{c}_boundary_recall_r{r}']=int(bt)
        out[f'class{c}_boundary_F1_r{r}']={'pred':int(pt),'GT':int(bt)}
    for k,g in enumerate(['small','tiny']):
        n,area=a[45+6*k:47+6*k]
        out[f'{g}_weed_pixel_recall']=int(area)
        out[f'{g}_weed_mean_patch_coverage']=int(n)
        out[f'{g}_weed_fraction_patches_at_least_half']=int(n)
    return out

def NA_reason(metric):
    if '_boundary_F1_' in metric:return 'BOTH_BOUNDARY_TOTALS_ZERO'
    if '_boundary_precision_' in metric:return 'PRED_BOUNDARY_TOTAL_ZERO'
    if '_boundary_recall_' in metric:return 'GT_BOUNDARY_TOTAL_ZERO'
    if '_weed_' in metric:return 'NO_GT_WEED_COMPONENT_IN_AREA_RANGE'
    return {'IoU':'CLASS_UNION_ZERO','precision':'PREDICTED_CLASS_TOTAL_ZERO','recall':'GT_CLASS_TOTAL_ZERO','F1':'GT_PLUS_PREDICTED_CLASS_TOTAL_ZERO'}[metric.split('_',1)[1]]

def fixed_five_summary(values):
    assert len(values)==5
    vs=[None if v is None or not math.isfinite(float(v)) else float(v) for v in values]
    if any(v is None for v in vs):
        return {'seed_values':vs,'n_defined':sum(v is not None for v in vs),'mean':None,'SD_ddof1':None,
                'min':None,'max':None,'range':None,'NA_reason':'ONE_OR_MORE_OF_FIXED_FIVE_SEEDS_UNDEFINED'}
    a=np.asarray(vs)
    return {'seed_values':vs,'n_defined':5,'mean':float(np.mean(a)),'SD_ddof1':float(np.std(a,ddof=1)),
            'min':float(np.min(a)),'max':float(np.max(a)),'range':float(np.max(a)-np.min(a)),'NA_reason':None}

def conditional_CI(values):
    a=np.asarray(values,dtype=np.float64);assert a.shape==(10000,)
    missing=int(np.count_nonzero(~np.isfinite(a)))
    if missing:return {'low':None,'high':None,'valid_draws':10000-missing,'NA_draws':missing,'NA_reason':'AT_LEAST_ONE_OF_FIXED_10000_DRAWS_UNDEFINED'}
    q=np.quantile(a,[.025,.975],method='linear')
    return {'low':float(q[0]),'high':float(q[1]),'valid_draws':10000,'NA_draws':0,'NA_reason':None}

def select_cases(records):
    """records contain fixed1701 unrounded values, not viewed RGB pixels."""
    groups=[('unfavourable_relative_to_specialist',lambda r:r['difference'] is not None and r['difference']<0,lambda r:(r['difference'],r['index'])),
            ('favourable_relative_to_specialist',lambda r:r['difference'] is not None and r['difference']>0,lambda r:(-r['difference'],r['index'])),
            ('small_patch_failure',lambda r:r['small_pixels']>0 and r['small_recall'] is not None,lambda r:(r['small_recall'],r['index']))]
    selected=[];audit=[];used_images=set();used_sessions=set();shortfalls=[]
    for name,eligible,sortkey in groups:
        good=sorted([dict(r) for r in records if eligible(r)],key=sortkey);picked=0
        for rank,r in enumerate(good,1):
            reason=('GROUP_QUOTA_ALREADY_FILLED' if picked>=2 else 'SAMPLE_ALREADY_SELECTED' if r['sample_id'] in used_images else
                    'SESSION_ALREADY_SELECTED' if r['session_id'] in used_sessions else None)
            row=dict(r,group=name,rank=rank,eligible=True,exclusion_reason=reason,selected=reason is None)
            audit.append(row)
            if reason is None:
                selected.append(row);picked+=1;used_images.add(r['sample_id']);used_sessions.add(r['session_id'])
        audit.extend(dict(r,group=name,rank=None,eligible=False,selected=False,exclusion_reason='UNDEFINED_OR_NOT_STRICT_SIGN_OR_NO_REQUIRED_GT_PATCH') for r in records if not eligible(r))
        shortfalls.append({'group':name,'requested':2,'selected':picked,'shortfall':2-picked})
    return selected,audit,shortfalls
