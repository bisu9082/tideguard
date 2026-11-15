import numpy as np, importlib.util, os, json
HERE="/sessions/epic-keen-bardeen/mnt/claude_research/tideguard"
spec=importlib.util.spec_from_file_location("wf",os.path.join(HERE,"wbe_forecast.py"))
wf=importlib.util.module_from_spec(spec); spec.loader.exec_module(wf)
CFG=wf.CFG
ser=wf.load_series()
funcs={"naive":wf.f_naive,"seasonal_naive":wf.f_seasonal_naive,"ets":wf.f_ets,
       "ar_ols":wf.f_ar_ols,"ridge_ar":wf.f_ridge_ar}
hmax=max(CFG["horizons"])
res={m:{"hit":0,"tot":0} for m in funcs}
for sid,w in ser.items():
    vals=np.asarray(w.values,float); n=len(vals)
    origins=[o for o in range(n-CFG["n_origins"]-hmax,n-hmax) if o>0]
    for hist in CFG["histories"]:
        for ms in CFG["missing_levels"]:
            for seed in CFG["seeds"]:
                rng=np.random.default_rng(seed)
                for o in origins:
                    b=vals[max(0,o-hist):o]
                    if len(b)<10: continue
                    bb=wf.fast_inject(b,ms,rng); last=bb[-1]
                    for h in CFG["horizons"]:
                        if o+h-1>=n: continue
                        yt=vals[o+h-1]
                        if np.sign(yt-last)==0: continue
                        for m,fn in funcs.items():
                            ph=fn(bb,h)
                            res[m]["tot"]+=1
                            if np.sign(ph-last)==np.sign(yt-last): res[m]["hit"]+=1
out={m:(round(r["hit"]/r["tot"],3) if r["tot"] else 0.0) for m,r in res.items()}
print(json.dumps(out,indent=2))
json.dump(out,open(os.path.join(HERE,"data","baseline_diracc.json"),"w"),indent=2)
