"""Richer step-0 features for the Frente A regime classifier (does OEP beat Producer this seed?)."""
import json, math, argparse
from pathlib import Path
from statistics import mean, pstdev
from scripts.p4_matrix import _run_match
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
from python.orbit_wars_gym.entities import planet_owner, planet_production, planet_x, planet_y, planet_ships
from python.orbit_wars_gym.rules import normalized_margin

def feats(state):
    P=state.get('planets',[]); homes=[p for p in P if planet_owner(p)>=0]; neut=[p for p in P if planet_owner(p)==-1]
    d=lambda a,b: math.hypot(planet_x(a)-planet_x(b),planet_y(a)-planet_y(b))
    xs=[planet_x(p) for p in P]; ys=[planet_y(p) for p in P]
    cx,cy=(mean(xs),mean(ys)) if P else (50,50)
    # inter-planet distances
    dd=[d(P[i],P[j]) for i in range(len(P)) for j in range(i+1,len(P))]
    # per-home: neutrals within radius, nearest-neutral, neutral prod within radius
    def near(h,R): return [nn for nn in neut if d(h,nn)<=R]
    f={'n_neutral':len(neut),'n_planets':len(P),
       'tot_neut_prod':sum(planet_production(nn) for nn in neut),
       'mean_neut_prod':mean([planet_production(nn) for nn in neut]) if neut else 0,
       'spread_x':round(pstdev(xs),1) if len(xs)>1 else 0,'spread_y':round(pstdev(ys),1) if len(ys)>1 else 0,
       'mean_interplanet':round(mean(dd),1) if dd else 0,'min_interplanet':round(min(dd),1) if dd else 0,
       'home_dist':round(d(homes[0],homes[1]),1) if len(homes)>=2 else 0,
       'home_prod_asym':abs(sum(planet_production(h) for h in homes[:1])-sum(planet_production(h) for h in homes[1:2])),
       }
    if len(homes)>=2:
        for R in (25,40):
            n0=len(near(homes[0],R)); n1=len(near(homes[1],R))
            f[f'neut_within_{R}_mean']=(n0+n1)/2; f[f'neut_within_{R}_asym']=abs(n0-n1)
        nn0=[d(homes[0],nn) for nn in neut]; f['nearest_neut_p0']=round(min(nn0),1) if nn0 else 0
    return f

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=64); ap.add_argument('--episode-steps',type=int,default=256)
    ap.add_argument('--out',default='artifacts/p4/features2.json'); a=ap.parse_args()
    prod=make_isolated_opponent('producer'); oep=make_isolated_opponent('oep'); rows=[]
    for seed in range(a.seeds):
        margins=[]
        for lineup,pidx in (([prod,oep],0),([oep,prod],1)):
            scores,per,steps,inv=_run_match(lineup,seed=seed,episode_steps=a.episode_steps,enable_comets=True,act_timeout=1.0)
            margins.append(normalized_margin(scores,pidx))
        pm=sum(margins)/2.0
        be=RustBatchBackend(num_envs=1,num_players=2,seed=seed,config=RustConfig(episode_steps=a.episode_steps,enable_comets=True))
        f=feats(be.reset(seed)[0]); rows.append({'seed':seed,'producer_margin':pm,'oep_favorable':pm<0,**f})
        print(f"seed {seed}: pm={pm:+.3f} oep_fav={pm<0}",flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(rows,indent=2))
    print(f"OEP-fav {sum(1 for r in rows if r['oep_favorable'])}/{len(rows)} saved {a.out}")
if __name__=='__main__': main()
