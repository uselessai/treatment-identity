#!/usr/bin/env python3
"""Check appendix sweep counts separately from its fixed-geometry counts."""
import argparse
import csv
import itertools
import json
from pathlib import Path
import fragility_derive as f

HERE=Path(__file__).resolve().parent


def derive():
    output={}
    for kernel,filename in [('area','geometry_fragility_v2_cells.csv'),('cubic','geometry_fragility_v2_cubic_cells.csv')]:
        _,data=f.load(HERE/filename)
        assert len(data)==27
        result={}
        for scale in (1.,2.):
            for metric in f.METRICS:
                decisive=endpoint=sweep=0
                for seed in ('2021','7','424242'):
                    keys=[k for k in data if k[-1]==seed];assert len(keys)==9
                    for a,b in itertools.combinations(keys,2):
                        delta=data[a][scale,0.][metric]-data[b][scale,0.][metric]
                        if abs(delta)<=f.TIE[metric]:continue
                        decisive+=1
                        points=[p for p in data[a] if p[0]==scale]
                        # The retained derivation keys realised geometry at six decimals.
                        end=(scale,round(f.INHERITED_D,6))
                        assert end in points
                        def inversion(p):
                            d=data[a][p][metric]-data[b][p][metric]
                            return abs(d)>f.TIE[metric] and delta*d<0
                        endpoint+=inversion(end)
                        sweep+=any(inversion(p) for p in points)
                result[f's{int(scale)}_{metric}']={'decisive_at_zero':decisive,'sweep_inversions':sweep,'inherited_geometry_inversions':endpoint}
        output[kernel]=result
    old=json.loads((HERE/'geometry_fragility_v2_derived.json').read_text())['by_scale_metric']
    for k,r in output['area'].items():
        assert r['decisive_at_zero']==old[k]['decisive_at_zero']
        assert r['sweep_inversions']==old[k]['inverted']
    assert output['area']['s2_LPIPS']=={'decisive_at_zero':106,'sweep_inversions':22,'inherited_geometry_inversions':22}
    assert output['cubic']['s2_LPIPS']['inherited_geometry_inversions']==2
    assert output['cubic']['s2_STLPIPS']['sweep_inversions']==9
    assert output['cubic']['s2_STLPIPS']['inherited_geometry_inversions']==0
    # Validate area endpoints for the four metrics available in the retained
    # independent endpoint table. This does not validate ST-LPIPS or cubic.
    frozen=list(csv.DictReader((HERE/'e18_geometry_sensitivity.csv').open()))
    cells=list(csv.DictReader((HERE/'geometry_fragility_v2_cells.csv').open()))
    checked=0;maximum={m:0. for m in ('PSNR','SSIM','LPIPS','DISTS')}
    for row in frozen:
        target=(640,368) if row['geometry']=='legacy' else (320,180)
        assert row['geometry'] in ('legacy','corrected')
        selected=[c for c in cells if all(c[k]==row[k] for k in ('arch','regime','seed'))
                  and (int(c['target_w']),int(c['target_h']))==target]
        assert selected
        for metric in maximum:
            error=max(abs(float(c[metric])-float(row[metric])) for c in selected)
            assert error<=(.005 if metric=='PSNR' else .0005)
            maximum[metric]=max(maximum[metric],error);checked+=1
    assert checked==216
    output['area_endpoint_validation']={'values_checked':checked,'maximum_absolute_error':maximum,
        'excluded':['STLPIPS','bicubic']}
    return output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path);args=ap.parse_args()
    result=derive()
    if args.out:args.out.write_text(json.dumps(result,indent=2)+'\n')
    print('Geometry derivation: raw-cell sweep counts and fixed-geometry counts verified separately.')


if __name__=='__main__':main()
