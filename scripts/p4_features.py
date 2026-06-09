"""Frente A: collect step-0 map features + per-seed Producer-vs-OEP margin (seat-neutral),
to learn a conservative selector (play OEP in OEP-favorable regimes, Producer else) that
could SURPASS Producer (always-producer only ties). Eval-only."""
import json, math, argparse
from pathlib import Path
from scripts.p4_matrix import _run_match
from python.agents.registry import make_isolated_opponent
from python.orbit_wars_gym.entities import planet_owner, planet_production, planet_x, planet_y
from python.orbit_wars_gym.rules import normalized_margin

def feats(state):
    P=state.get('planets',[]); homes=[p for p in P if planet_owner(p)>=0]; neut=[p for p in P if planet_owner(p)==-1]
    d=lambda a,b: math.hypot(planet_x(a)-planet_x(b),planet_y(a)-planet_y(b))
    nn=[min(d(h,n) for n in neut) for h in homes if neut]
    return {'n_neutral':len(neut),'home_dist':round(d(homes[0],homes[1]),1) if len(homes)>=2 else 0.0,
            'mean_nn':round(sum(nn)/len(nn),1) if nn else 0.0,'tot_neut_prod':sum(planet_production(n) for n in neut),
            'n_planets':len(P)}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seeds',type=int,default=48); ap.add_argument('--episode-steps',type=int,default=256)
    ap.add_argument('--out',default='artifacts/p4/features_labels.json'); a=ap.parse_args()
    prod=make_isolated_opponent('producer'); oep=make_isolated_opponent('oep')
    rows=[]
    for seed in range(a.seeds):
        # both seatings -> seat-neutral producer margin (producer as p0 then p1)
        margins=[]
        f=None
        for lineup,pidx,oidx in (([prod,oep],0,1),([oep,prod],1,0)):
            scores,per,steps,inv=_run_match(lineup,seed=seed,episode_steps=a.episode_steps,enable_comets=True,act_timeout=1.0)
            margins.append(normalized_margin(scores,pidx))  # producer's margin this seating
        prod_margin=sum(margins)/2.0  # seat-neutral producer-vs-oep margin (>0 producer favored)
        # features from step-0 (seating-independent; reset once)
        from python.orbit_wars_gym.backend import RustBatchBackend, RustConfig
        be=RustBatchBackend(num_envs=1,num_players=2,seed=seed,config=RustConfig(episode_steps=a.episode_steps,enable_comets=True))
        f=feats(be.reset(seed)[0])
        oep_fav = prod_margin < 0  # OEP beat Producer this seed
        rows.append({'seed':seed,'producer_margin':prod_margin,'oep_favorable':oep_fav,**f})
        print(f"seed {seed}: prod_margin={prod_margin:+.3f} oep_fav={oep_fav} neut={f['n_neutral']} neut_prod={f['tot_neut_prod']}",flush=True)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(rows,indent=2))
    nfav=sum(1 for r in rows if r['oep_favorable']); print(f"\nOEP-favorable: {nfav}/{len(rows)} | saved {a.out}")

if __name__=='__main__': main()
