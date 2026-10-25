#!/usr/bin/env python3
"""Complete finite double-sweep spectra. No source solve/CG trajectory.

Construct both sweeps explicitly, then independently test Qhat=Fhat*Fhat.
Preserve structural replica supports. Do not discard zero stored coefficients.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key]=os.environ.get('WCARP_THREADS','1')
from pathlib import Path
import sys,json,time,ctypes,hashlib
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
LIB=ctypes.CDLL(str(Path(__file__).with_name('row_sweeps.so')))
PTR=ctypes.c_void_p
LIB.sweep_rows.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.c_int,PTR,PTR,PTR,PTR,PTR,ctypes.c_double,ctypes.c_int,PTR,PTR]
LIB.sweep_rows.restype=ctypes.c_int

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relative(x,y): return float(la.norm(x-y)/max(la.norm(y),np.finfo(float).tiny))
def jwrite(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2));tmp.replace(p)

def sweep(d,x,reverse=False,with_loss=False):
    x=np.ascontiguousarray(x,dtype=np.complex128)
    if x.ndim==1: x=x[:,None]
    iw=np.ascontiguousarray(1/d['w'])
    loss=np.zeros(x.shape[1]) if with_loss else None
    rc=LIB.sweep_rows(len(d['row_gids']),len(d['col_gids']),x.shape[1],
        d['indptr'].ctypes.data,d['indices'].ctypes.data,d['data'].ctypes.data,
        iw.ctypes.data,d['inv_row_denom'].ctypes.data,float(d['lambda']),int(reverse),
        x.ctypes.data,None if loss is None else loss.ctypes.data)
    assert rc==0,rc
    return x,loss

def load(case,beta):
    token={0:'beta0',.5:'beta0p5',1:'beta1'}[beta]
    run=ROOT/'exporter/runs'/f'{case}_{token}'
    paths=sorted(run.glob('operator_rank*.npz'))
    assert len(paths)==8,(run,paths)
    ds=[]
    for p in paths:
        with np.load(p) as f: d={k:np.ascontiguousarray(f[k]) for k in f.files}
        # Reader converts zero-based index arrays but retains int32 storage.
        for k in ['row_gids','col_gids','indices','indptr']:
            d[k]=np.ascontiguousarray(d[k],dtype=np.int32)
        ds.append(d)
    return run,ds

def global_setup(ds,beta):
    n=int(ds[0]['nglobal']); m=np.zeros(n)
    allrows=np.concatenate([d['row_gids'] for d in ds])
    assert np.array_equal(np.sort(allrows),np.arange(n))
    rr=[];cc=[];aa=[];raw=[];fullprobe=np.zeros((n,3),complex)
    eps=1e-8*max(float(d['chi'].max()) for d in ds)
    checks={'weight_formula_max_relative':0.,'metric_max_relative':0.,'row_denom_max_relative':0.,'scalar_floors_triggered':0}
    for d in ds:
        gids=d['col_gids'];expected=(d['chi']+eps)**beta
        checks['weight_formula_max_relative']=max(checks['weight_formula_max_relative'],float(np.max(np.abs(d['w']-expected)/expected)))
        np.add.at(m,gids,d['w'])
        for i,rowid in enumerate(d['row_gids']):
            sl=slice(d['indptr'][i],d['indptr'][i+1]);cols=d['indices'][sl];v=d['data'][sl]
            den=float(np.sum(np.abs(v)**2/d['w'][cols]))
            checks['scalar_floors_triggered']+=int(den<1e-12)
            checks['row_denom_max_relative']=max(checks['row_denom_max_relative'],abs(d['inv_row_denom'][i]-1/max(den,1e-12))/(1/max(den,1e-12)))
            rr.extend([int(rowid)]*len(v));cc.extend(gids[cols].tolist());aa.extend(v);raw.extend(d['raw_data'][sl])
        fullprobe[gids]=d['probe_in']
    assert np.all(m>0)
    for d in ds:
        checks['metric_max_relative']=max(checks['metric_max_relative'],float(np.max(np.abs(d['m']-m[d['col_gids']])/m[d['col_gids']])))
        assert relative(fullprobe[d['col_gids']],d['probe_in'])<1e-14
        checks['scalar_floors_triggered']+=int(np.count_nonzero(d['w']<1e-12))+int(np.count_nonzero(d['m']<1e-12))
    a=sp.coo_matrix((aa,(rr,cc)),shape=(n,n)).tocsr()
    ar=sp.coo_matrix((raw,(rr,cc)),shape=(n,n)).tocsr()
    assert max(checks[k] for k in checks if k.endswith('relative'))<2e-12,checks
    assert checks['scalar_floors_triggered']==0,checks
    return m,a,ar,fullprobe,checks

def dense_sweeps(ds,m,probe):
    n=len(m);sm=np.sqrt(m);f=np.zeros((n,n),complex)
    checks={'local_forward_relative':0.,'forward_consensus_relative':0.,'local_backward_relative':0.,'double_sweep_relative':0.,'dispatcher_relative':0.,'original_A_probe_relative':0.}
    z=sm[:,None]*probe
    for d in ds:
        gids=d['col_gids']; x=np.zeros((len(gids),n),complex)
        x[np.arange(len(gids)),gids]=1/sm[gids]
        x,_=sweep(d,x)
        checks['local_forward_relative']=max(checks['local_forward_relative'],relative(x@z,d['local_forward']))
        f[gids]+=d['w'][:,None]*x/sm[gids,None]
    q=np.zeros_like(f)
    for d in ds:
        gids=d['col_gids']; x=np.ascontiguousarray(f[gids]/sm[gids,None])
        checks['forward_consensus_relative']=max(checks['forward_consensus_relative'],relative(x@z,d['forward_consensus']))
        x,_=sweep(d,x,reverse=True)
        checks['local_backward_relative']=max(checks['local_backward_relative'],relative(x@z,d['local_backward']))
        q[gids]+=d['w'][:,None]*x/sm[gids,None]
    qp=(q@z)/sm[:,None]
    for d in ds:
        gids=d['col_gids']
        checks['double_sweep_relative']=max(checks['double_sweep_relative'],relative(qp[gids],d['double_sweep']))
        checks['dispatcher_relative']=max(checks['dispatcher_relative'],relative(qp[gids],d['dispatcher']))
    h=-q;h[np.diag_indices(n)]+=1
    checks['hermitian_relative']=relative(h,h.conj().T)
    checks['factorization_relative']=relative(q,f.conj().T@f)
    assert max(checks.values())<2e-11,checks
    return h,f,checks

def row_energy(ds,m,e):
    e=np.asarray(e); e=e[:,None] if e.ndim==1 else e
    forward=[]; loss=np.zeros(e.shape[1]); avg=np.zeros(e.shape,complex)
    for d in ds:
        x,l=sweep(d,e[d['col_gids']].copy(),with_loss=True);forward.append(x);loss+=l
        avg[d['col_gids']]+=d['w'][:,None]*x/m[d['col_gids'],None]
    disagreement=np.zeros_like(loss)
    for d,x in zip(ds,forward):disagreement+=np.sum(d['w'][:,None]*np.abs(x-avg[d['col_gids']])**2,axis=0)
    denom=np.sum(m[:,None]*np.abs(e)**2,axis=0)
    return loss/denom,disagreement/denom

def analyze(case,beta):
    token={0:'beta0',.5:'beta0p5',1:'beta1'}[beta];out=ROOT/'results/spectra'/case/token
    if (out/'RESULT.json').exists():
        d=json.loads((out/'RESULT.json').read_text());assert d['status']=='COMPLETE';print('REUSE',case,token,flush=True);return
    out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    jwrite(out/'STATUS.json',{'state':'RUNNING','pid':os.getpid(),'case':case,'beta':beta})
    run,ds=load(case,beta);m,a,ar,probe,setup=global_setup(ds,beta)
    print('BUILD',case,token,'n',len(m),flush=True)
    h,f,checks=dense_sweeps(ds,m,probe)
    ap=a@probe
    checks['original_A_probe_relative']=max(relative(ap[d['row_gids']],d['matvec']) for d in ds)
    assert checks['original_A_probe_relative']<2e-12
    hs=(h+h.conj().T)/2
    vals,vecs=la.eigh(hs,driver='evd',check_finite=False,overwrite_a=False)
    residual=h@vecs-vecs*vals[None,:]
    eig_abs=np.sqrt(np.sum(abs(residual)**2,axis=0))
    checks['eigendecomposition_fro_relative']=float(la.norm(residual)/la.norm(h))
    checks['low40_max_eigenpair_absolute_residual']=float(eig_abs[:40].max())
    checks['max_eigenpair_absolute_residual']=float(eig_abs.max())
    checks['orthogonality_fro_relative']=float(la.norm(vecs.conj().T@vecs-np.eye(len(m)))/np.sqrt(len(m)))
    uncert=max(float(eig_abs.max()),np.finfo(float).eps*len(m)*float(la.norm(h,np.inf)))
    threshold=10*uncert;positive=vals[vals>threshold]
    nneg=int(np.count_nonzero(vals < -threshold));nzero=int(np.count_nonzero(abs(vals)<=threshold))
    assert nneg==0 and vals[-1]<1+threshold
    e=vecs[:,:40]/np.sqrt(m[:,None]);row,copy=row_energy(ds,m,e)
    checks['low40_energy_identity_absolute']=float(np.max(np.abs(row+copy-vals[:40])))
    rowp,copyp=row_energy(ds,m,probe)
    hp=(h@(np.sqrt(m[:,None])*probe))/np.sqrt(m[:,None])
    ray=np.real(np.sum(probe.conj()*m[:,None]*hp,axis=0)/np.sum(abs(probe)**2*m[:,None],axis=0))
    checks['probe_energy_identity_absolute']=float(np.max(np.abs(rowp+copyp-ray)))
    assert checks['eigendecomposition_fro_relative']<1e-11 and checks['orthogonality_fro_relative']<1e-11
    assert checks['low40_energy_identity_absolute']<1e-11 and checks['probe_energy_identity_absolute']<1e-11
    np.save(out/'Hhat.npy',h);np.save(out/'Fhat.npy',f);np.save(out/'M.npy',m)
    np.save(out/'eigenvalues.npy',vals);np.save(out/'low_eigenvectors.npy',vecs[:,:40])
    np.savez(out/'low_energy.npz',row_loss=row,copy_disagreement=copy)
    sp.save_npz(out/'A_scaled.npz',a);sp.save_npz(out/'A_raw.npz',ar)
    result={'status':'COMPLETE','case':case,'beta':beta,'n':len(m),'eigenvalue_method':'All eigenvalues and eigenvectors from complex128 dense Hermitian LAPACK divide-and-conquer; direct un-symmetrized operator residuals checked','min_eigenvalue':float(vals[0]),'max_eigenvalue':float(vals[-1]),'lambda_min_positive':float(positive[0]) if len(positive) else None,'kappa_positive':float(positive[-1]/positive[0]) if len(positive) else None,'minimum_eigenvalue_resolved':bool(vals[0]>threshold),'numerical_uncertainty_scale':uncert,'positive_threshold':threshold,'negative_count':nneg,'unresolved_or_zero_count':nzero,'counts_below':{str(t):int(np.count_nonzero(vals<t)) for t in [1e-6,1e-5,1e-4,1e-3,1e-2,.1]},'M_range':[float(m.min()),float(m.max())],'copy_count':int(sum(len(d['col_gids']) for d in ds)),'setup_checks':setup,'operator_checks':checks,'low20_energy':{'row_mean':float(row[:20].mean()),'copy_mean':float(copy[:20].mean())},'export_directory':str(run),'elapsed_s':time.monotonic()-start,'artifacts':{p.name:sha(p) for p in out.iterdir() if p.suffix in ['.npy','.npz']}}
    jwrite(out/'RESULT.json',result);jwrite(out/'STATUS.json',{'state':'COMPLETE','elapsed_s':result['elapsed_s']})
    print('DONE',case,token,'lambda_min',vals[0],'kappa',result['kappa_positive'],'seconds',result['elapsed_s'],flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=int(os.environ.get('WCARP_THREADS','1'))):
        plan=json.loads((ROOT/'PLAN.json').read_text())
        selected=sys.argv[1:]
        for case in plan['cases']:
            if selected and case['id'] not in selected:continue
            for beta in plan['beta']:analyze(case['id'],beta)
