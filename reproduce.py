#!/usr/bin/env python3
"""Portable checks and small-system WCARP-CG demonstration.

Inputs are the fixed exported numerical operators from the manuscript's four
spectral tests. No external geological model or Intel MPI installation is needed.
"""
import argparse
import csv
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = os.environ.get('WCARP_THREADS', '1')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'analysis'))
from tools.build import build
build()
import numpy as np
import scipy.sparse.linalg as sla
from threadpoolctl import threadpool_limits
import analyze_spectrum as archive

PLAN = json.loads((ROOT / 'PLAN.json').read_text())
CASES = [c['id'] for c in PLAN['cases']]
TOKENS = {0.0: 'beta0', 0.5: 'beta0p5', 1.0: 'beta1'}

archive.LIB.sweep_affine.argtypes = [ctypes.c_int, ctypes.c_int] + [ctypes.c_void_p]*5 + [ctypes.c_double, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
archive.LIB.sweep_affine.restype = ctypes.c_int

class System:
    def __init__(self, case, beta):
        self.case, self.beta = case, beta
        _, self.blocks = archive.load(case, beta)
        self.m, self.a, self.raw_a, self.probe, self.checks = archive.global_setup(self.blocks, beta)
        self.n = len(self.m)
        self.b = np.zeros(self.n, dtype=np.complex128)
        for d in self.blocks:
            self.b[d['row_gids']] = d['rhs']
            d['invw'] = np.ascontiguousarray(1 / d['w'])

    def half(self, x, reverse=False, affine=False):
        x = np.asarray(x, dtype=np.complex128)
        out = np.zeros_like(x)
        for d in self.blocks:
            gids = d['col_gids']
            local = np.ascontiguousarray(x[gids])
            if affine:
                if x.ndim != 1:
                    raise ValueError('Affine application expects one vector')
                rc = archive.LIB.sweep_affine(len(d['row_gids']), len(gids),
                    d['indptr'].ctypes.data, d['indices'].ctypes.data,
                    d['data'].ctypes.data, d['invw'].ctypes.data,
                    d['inv_row_denom'].ctypes.data, float(d['lambda']), int(reverse),
                    d['rhs'].ctypes.data, local.ctypes.data)
                if rc:
                    raise RuntimeError(f'Row kernel error {rc}')
            else:
                local, _ = archive.sweep(d, local, reverse)
                if x.ndim == 1:
                    local = local[:, 0]
            scale = d['w'] / self.m[gids]
            out[gids] += (scale if x.ndim == 1 else scale[:, None]) * local
        return out

    def double(self, x, affine=False):
        return self.half(self.half(x, affine=affine), reverse=True, affine=affine)

    def residual(self, x):
        return float(np.linalg.norm(self.b - self.a @ x) / np.linalg.norm(self.b))

def quick_check():
    provenance = json.loads((ROOT / 'PROVENANCE.json').read_text())
    for item in provenance['export_files']:
        assert hashlib.sha256((ROOT / item['file']).read_bytes()).hexdigest() == item['sha256'], item['file']
    metrics = list(csv.DictReader((ROOT / 'paper_figure/spectral_metrics.csv').open()))
    records = []
    for case in CASES:
        baseline = None
        for beta in (0.0, 0.5, 1.0):
            s = System(case, beta)
            if baseline is None:
                baseline = (s.a.copy(), s.b.copy())
            delta = s.a - baseline[0]
            assert delta.nnz == 0 or np.max(np.abs(delta.data)) < 1e-14
            assert np.array_equal(s.b, baseline[1])
            forward = s.half(s.probe)
            double = s.half(forward, reverse=True)
            action_error = max(archive.relative(double[d['col_gids']], d['dispatcher']) for d in s.blocks)
            forward_error = max(archive.relative(forward[d['col_gids']], d['forward_consensus']) for d in s.blocks)
            assert action_error < 2e-11 and forward_error < 2e-11
            # Independently compare the affine kernel to a direct Python row loop.
            d = s.blocks[0]
            actual = np.ascontiguousarray(s.probe[d['col_gids'], 0])
            expected = actual.copy()
            for row in range(len(d['row_gids'])):
                sl = slice(d['indptr'][row], d['indptr'][row+1])
                cols, a = d['indices'][sl], d['data'][sl]
                residual = d['rhs'][row] - a @ expected[cols]
                expected[cols] += float(d['lambda']) * d['inv_row_denom'][row] * a.conj() * residual / d['w'][cols]
            rc = archive.LIB.sweep_affine(len(d['row_gids']), len(d['col_gids']),
                d['indptr'].ctypes.data, d['indices'].ctypes.data,
                d['data'].ctypes.data, d['invw'].ctypes.data, d['inv_row_denom'].ctypes.data,
                float(d['lambda']), 0, d['rhs'].ctypes.data, actual.ctypes.data)
            assert rc == 0 and archive.relative(actual, expected) < 2e-11
            refdir = ROOT / 'reference' / case / TOKENS[beta]
            vals = np.load(refdir / 'eigenvalues.npy', allow_pickle=False)
            saved = json.loads((refdir / 'RESULT.json').read_text())
            row = next(r for r in metrics if r['case'] == case and float(r['beta']) == beta)
            assert len(vals) == s.n and np.all(np.isfinite(vals))
            assert np.isclose(float(row['kappa']), saved['kappa_positive'], rtol=1e-10)
            assert np.isclose(vals[-1]/vals[0], saved['kappa_positive'], rtol=1e-10)
            records.append({'case': case, 'beta': beta, 'unknowns': s.n,
                'double_sweep_relative_difference': action_error,
                'forward_relative_difference': forward_error,
                'affine_kernel_relative_difference': archive.relative(actual, expected),
                'archived_kappa_positive': saved['kappa_positive']})
    out = ROOT / 'results'; out.mkdir(exist_ok=True)
    (out / 'quick_check.json').write_text(json.dumps({'passed': True, 'checks': records}, indent=2)+'\n')
    print('PASS: 96 input hashes; all 12 weighted operators; common A,b; affine kernel; archived spectrum/figure values.', flush=True)

def demo(case, beta, tol, maxiter):
    s = System(case, beta)
    direct = sla.spsolve(s.a.tocsc(), s.b)
    sm = np.sqrt(s.m)
    g = sm * s.double(np.zeros(s.n, complex), affine=True)
    def action(y):
        x = y / sm
        return sm * (x - s.double(x))
    y = np.zeros(s.n, complex)
    r = g.copy(); p = r.copy(); rr = float(np.vdot(r, r).real)
    history = [{'iteration': 0, 'relative_row_scaled_residual': 1.0}]
    for k in range(1, maxiter+1):
        hp = action(p)
        php = np.vdot(p, hp)
        if not np.isfinite(php) or php.real <= 0 or abs(php.imag) > 1e-8*abs(php.real):
            raise RuntimeError(f'Invalid CG curvature at iteration {k}: {php}')
        alpha = rr/php.real
        y += alpha*p; r -= alpha*hp
        residual = s.residual(y/sm)
        history.append({'iteration': k, 'relative_row_scaled_residual': residual})
        if not np.isfinite(residual):
            raise RuntimeError('Nonfinite original-system residual')
        if residual <= tol:
            break
        rr_new = float(np.vdot(r, r).real)
        if rr_new <= 0 or not np.isfinite(rr_new):
            raise RuntimeError('CG residual breakdown')
        p = r + (rr_new/rr)*p; rr = rr_new
    if residual > tol:
        raise RuntimeError(f'Iteration cap reached: {residual} > {tol}')
    x = y/sm
    result = {'case': case, 'beta': beta, 'iterations': k, 'unknowns': s.n,
        'row_scaled_relative_residual': residual,
        'direct_solution_relative_residual': s.residual(direct),
        'relative_complex_field_difference_to_direct': float(np.linalg.norm(x-direct)/np.linalg.norm(direct)),
        'method': 'CG in the induced M metric, implemented by the equivalent similarity transform',
        'scope': 'Portable small-system reproduction demonstration; not a published production timing result.'}
    out = ROOT / 'results/demo' / case / TOKENS[beta]; out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'solutions.npz', iterative=x, direct=direct, rhs=s.b, metric=s.m)
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'history.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    print(json.dumps(result,indent=2),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check','demo','spectrum'])
    parser.add_argument('--case', choices=CASES, default=CASES[0])
    parser.add_argument('--beta', type=float, choices=[0,.5,1], default=.5)
    parser.add_argument('--tol', type=float, default=1e-8)
    parser.add_argument('--maxiter', type=int, default=5000)
    args=parser.parse_args()
    if not 0 < args.tol < 1 or args.maxiter < 1:
        parser.error('Require 0 < tol < 1 and maxiter >= 1')
    with threadpool_limits(limits=int(os.environ.get('WCARP_THREADS','1'))):
        if args.command == 'check': quick_check()
        elif args.command == 'demo': demo(args.case,args.beta,args.tol,args.maxiter)
        else:
            archive.analyze(args.case,args.beta)
            token = TOKENS[args.beta]
            got = np.load(ROOT/'results/spectra'/args.case/token/'eigenvalues.npy')
            reference = np.load(ROOT/'reference'/args.case/token/'eigenvalues.npy')
            error = float(np.max(np.abs(got-reference)))
            assert error < 2e-11, error
            print('PASS: full spectrum agrees with archived values; maximum absolute difference', error)
