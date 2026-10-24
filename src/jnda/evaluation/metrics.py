"""Class, reliable boundary and small weed-patch statistics."""
from collections import Counter
import math
import numbers
import numpy as np
from scipy import ndimage

CLASSES={'crop':1,'weed':2}
TOLERANCES=(1,2,4)
INT64_MAX=(1<<63)-1

def require(value,reason):
    if not value:raise ValueError(reason)

def labels(prediction,target):
    p=np.asarray(prediction);g=np.asarray(target)
    require(p.ndim==g.ndim==2 and p.shape==g.shape and all(n>0 for n in p.shape),'same_nonempty_2D_label_shape_required')
    require(np.issubdtype(p.dtype,np.integer) and np.issubdtype(g.dtype,np.integer),'integer_category_arrays_required')
    require(bool(np.isin(p,[0,1,2]).all()),'invalid_prediction_category_no_void_prediction')
    require(bool(np.isin(g,[0,1,2,255]).all()),'invalid_GT_category_or_class_mapping')
    return p,g,g!=255

def disk_offsets(radius):
    require(type(radius) is int and radius in TOLERANCES,'only_predeclared_pixel_tolerances_1_2_4')
    return [(dy,dx) for dy in range(-radius,radius+1) for dx in range(-radius,radius+1)
            if dy*dy+dx*dx<=radius*radius]

def disk(radius):
    a=np.zeros((2*radius+1,2*radius+1),dtype=bool)
    for y,x in disk_offsets(radius):a[y+radius,x+radius]=True
    return a

def slices(shape,dy,dx):
    h,w=shape
    if abs(dy)>=h or abs(dx)>=w:
        return ((slice(0,0),slice(0,0)),(slice(0,0),slice(0,0)))
    return ((slice(max(0,-dy),min(h,h-dy)),slice(max(0,-dx),min(w,w-dx))),
            (slice(max(0,dy),min(h,h+dy)),slice(max(0,dx),min(w,w+dx))))

def inner_contour(mask,valid):
    """Foreground pixel with a known valid nonclass 4-neighbour only."""
    result=np.zeros(mask.shape,dtype=bool)
    for dy,dx in ((-1,0),(1,0),(0,-1),(0,1)):
        a,b=slices(mask.shape,dy,dx)
        result[a] |= mask[a]&valid[a]&valid[b]&~mask[b]
    return result

def directional_matches(pred_boundary,gt_boundary,components,radius):
    """Any-within-tolerance coverage in each direction, not one-to-one pairs.

    Public counts pass eroded-domain contours. Thus each retained centre's
    entire pixel-centre radius-r disk is valid. Any matched centre is in that
    disk, so no ignore hole can provide an intervening valid match shortcut.
    Original valid-domain8-component equality is enforced independently here.
    """
    mp=np.zeros(pred_boundary.shape,dtype=bool);mg=np.zeros_like(mp)
    for dy,dx in disk_offsets(radius):
        a,b=slices(mp.shape,dy,dx)
        pair=pred_boundary[a]&gt_boundary[b]&(components[a]>0)&(components[a]==components[b])
        mp[a] |= pair;mg[b] |= pair
    return int(mp.sum()),int(mg.sum())

def boundary_scores(pred_total,gt_total,pred_matched,gt_matched):
    for v in (pred_total,gt_total,pred_matched,gt_matched):
        require(isinstance(v,numbers.Integral) and not isinstance(v,(bool,np.bool_)) and 0<=v<=INT64_MAX,'nonnegative_int64_counts_required')
    require(pred_matched<=pred_total and gt_matched<=gt_total,'directional_matches_exceed_denominator')
    if pred_total==gt_total==0:
        return {'precision':None,'recall':None,'f1':None,'status':'NA','reason':'BOTH_BOUNDARY_TOTALS_ZERO'}
    precision=pred_matched/pred_total if pred_total else None
    recall=gt_matched/gt_total if gt_total else None
    if not pred_total or not gt_total:
        return {'precision':precision,'recall':recall,'f1':0.,'status':'DEFINED_ONE_SIDED_EMPTY',
                'reason':'PRED_BOUNDARY_EMPTY' if not pred_total else 'GT_BOUNDARY_EMPTY'}
    f1=2*precision*recall/(precision+recall) if precision+recall else 0.
    return {'precision':precision,'recall':recall,'f1':f1,'status':'DEFINED','reason':None}

def boundary_counts(prediction,target,class_id,radius):
    require(type(class_id) is int and class_id in CLASSES.values(),'only_crop_or_weed_boundary')
    disk_offsets(radius);p,g,valid=labels(prediction,target)
    safe=ndimage.binary_erosion(valid,structure=disk(radius),border_value=0)
    components,ncomponents=ndimage.label(valid,structure=np.ones((3,3),dtype=bool))
    pb=inner_contour(p==class_id,valid);gb=inner_contour(g==class_id,valid)
    rawp,rawg=int(pb.sum()),int(gb.sum());pb &=safe;gb &=safe
    pt,gt=int(pb.sum()),int(gb.sum());pm,gm=directional_matches(pb,gb,components,radius)
    reason=('EMPTY_VALID_DOMAIN' if not valid.any() else 'EMPTY_ERODED_VALID_DOMAIN' if not safe.any()
            else 'NO_RELIABLE_BOUNDARIES' if not pt and not gt else None)
    return {'schema':'jnda-e10-boundary-directional-counts-v1','class_id':class_id,'radius_pixels':radius,
      'pred_boundary_total':pt,'gt_boundary_total':gt,'pred_matched':pm,'gt_matched':gm,
      'valid_pixels':int(valid.sum()),'eroded_valid_pixels':int(safe.sum()),
      'original_valid_components_8':int(ncomponents),'raw_pred_contours':rawp,'raw_gt_contours':rawg,
      'excluded_pred_contours':rawp-pt,'excluded_gt_contours':rawg-gt,'image_NA_reason':reason,
      'scores':boundary_scores(pt,gt,pm,gm)}

def confusion_from_synthetic_arrays(prediction,target):
    p,g,valid=labels(prediction,target)
    return np.bincount((g[valid].astype(np.int64)*3+p[valid]).ravel(),minlength=9).reshape(3,3).astype(np.int64).tolist()

def validate_boundary_record(row):
    require(row['class_id'] in (1,2) and row['radius_pixels'] in TOLERANCES,'declared_class_and_radius_required')
    names=('pred_boundary_total','gt_boundary_total','pred_matched','gt_matched')
    boundary_scores(*(row[k] for k in names))
    for k in ('valid_pixels','eroded_valid_pixels','original_valid_components_8','raw_pred_contours','raw_gt_contours',
              'excluded_pred_contours','excluded_gt_contours'):
        require(type(row[k]) is int and 0<=row[k]<=INT64_MAX,'int64_boundary_auxiliary_counts')
    require(row['eroded_valid_pixels']<=row['valid_pixels'] and
            row['raw_pred_contours']==row['pred_boundary_total']+row['excluded_pred_contours'] and
            row['raw_gt_contours']==row['gt_boundary_total']+row['excluded_gt_contours'],
            'boundary_auxiliary_count_consistency')
    return True

def aggregate_boundary(rows):
    require(bool(rows),'at_least_one_image_count_record_required')
    key=(rows[0]['class_id'],rows[0]['radius_pixels'])
    require(all((r['class_id'],r['radius_pixels'])==key for r in rows),'do_not_mix_class_or_tolerance')
    for r in rows:validate_boundary_record(r)
    totals={name:sum(int(r[name]) for r in rows) for name in
            ('pred_boundary_total','gt_boundary_total','pred_matched','gt_matched','excluded_pred_contours','excluded_gt_contours')}
    scores=boundary_scores(totals['pred_boundary_total'],totals['gt_boundary_total'],totals['pred_matched'],totals['gt_matched'])
    reasons=Counter(r['image_NA_reason'] for r in rows if r['image_NA_reason'])
    return {'class_id':key[0],'radius_pixels':key[1],**totals,'scores':scores,'images':len(rows),
      'NA_image_reasons':dict(sorted(reasons.items())),'both_empty_images':sum(r['pred_boundary_total']==r['gt_boundary_total']==0 for r in rows),
      'one_sided_empty_images':sum((r['pred_boundary_total']==0)!=(r['gt_boundary_total']==0) for r in rows),
      'aggregation':'sum int64 directional counts before ratios; not mean image F1'}

def weed_patch_records(prediction,target,*,sample_id,source,session_id):
    p,g,valid=labels(prediction,target)
    require(all(isinstance(v,str) and v for v in (sample_id,source,session_id)),'exact_sample_source_session_metadata_required')
    components,n=ndimage.label((g==2)&valid,structure=np.ones((3,3),dtype=bool))
    areas=np.bincount(components.ravel(),minlength=n+1)
    covered=np.bincount(components[p==2],minlength=n+1)
    records=[]
    for i in range(1,n+1):
        area=int(areas[i]);hits=int(covered[i]);require(0<=hits<=area,'patch_overlap_denominator')
        if area<=256:
            records.append({'sample_id':sample_id,'source':source,'session_id':session_id,'component_id':i,
              'area_pixels':area,'covered_pixels':hits,'coverage':hits/area,'tiny':area<=16,
              'coverage_at_least_half':2*hits>=area})
    return {'schema':'jnda-e10-small-weed-patches-v1','sample_id':sample_id,'source':source,'session_id':session_id,
      'all_GT_weed_components':int(n),'excluded_components_above256':int(sum(areas[1:]>256)),
      'valid_pixels':int(valid.sum()),'records':records,'image_has_small':bool(records),
      'image_has_tiny':any(r['tiny'] for r in records)}

def aggregate_patches(images,multiplicities=None):
    require(bool(images),'patch_image_records_required')
    ids=[r['sample_id'] for r in images];require(len(ids)==len(set(ids)),'duplicate_patch_image')
    for image in images:
        components=[r['component_id'] for r in image['records']]
        require(len(components)==len(set(components)),'duplicate_patch_component')
        for r in image['records']:
            a,h=r['area_pixels'],r['covered_pixels']
            require(type(a) is int and type(h) is int and 1<=a<=256 and 0<=h<=a and
                    r['coverage']==h/a and r['tiny']==(a<=16) and r['coverage_at_least_half']==(2*h>=a),
                    'patch_count_or_denominator_record_changed')
            require(all(r[k]==image[k] for k in ('sample_id','source','session_id')),'patch_source_session_or_image_changed')
    multiplicities={sid:1 for sid in ids} if multiplicities is None else multiplicities
    require(set(multiplicities)==set(ids) and all(type(w) is int and w>=0 for w in multiplicities.values()),'whole_image_integer_resampling_weights')
    sessions={(r['source'],r['session_id']) for r in images};answer={}
    for name,maximum in [('small',256),('tiny',16)]:
        records=[{**r,'session_draw_multiplicity':multiplicities[image['sample_id']]} for image in images
                 for r in image['records'] if r['area_pixels']<=maximum and multiplicities[image['sample_id']]>0]
        area=sum(r['area_pixels']*r['session_draw_multiplicity'] for r in records)
        hits=sum(r['covered_pixels']*r['session_draw_multiplicity'] for r in records)
        n=sum(r['session_draw_multiplicity'] for r in records)
        qualified=sum(r['coverage_at_least_half']*r['session_draw_multiplicity'] for r in records)
        require(area<=INT64_MAX and hits<=area,'pooled_patch_int64_denominators')
        answer[name]={'patches':n,'pixels':area,'covered_pixels':hits,'images_with_patches':len({r['sample_id'] for r in records}),
          'sessions_with_patches':len({(r['source'],r['session_id']) for r in records}),
          'pixel_recall':hits/area if area else None,'mean_patch_coverage':math.fsum(r['coverage']*r['session_draw_multiplicity'] for r in records)/n if n else None,
          'patches_at_least_half':qualified,'fraction_patches_at_least_half':qualified/n if n else None,
          'status':'DEFINED' if n else 'NA','reason':None if n else 'NO_GT_WEED_COMPONENT_IN_AREA_RANGE',
          'area_range_pixels':[1,maximum],'patch_records':records}
    return {'images_evaluated':len(images),'image_occurrences_in_session_draw':sum(multiplicities.values()),
      'sessions_evaluated':len(sessions),'groups':answer,
      'uncertainty_unit':'complete source/session, never independent patch bootstrap',
      'not_biological_instances_or_AP':True}
