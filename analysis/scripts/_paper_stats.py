"""Compute all paper-facing statistics for results.tex."""
import sys
sys.path.insert(0, 'analysis/scripts')
from _leakage_common import load_data, ATTR_FAMILIES, APP_CATEGORY, CHANNELS
import pandas as pd

df, raw = load_data()
flt = df.attrs["filter"]

print("=== CORPUS ===")
print(f"items={df['full_key'].nunique()}, pairs={len(df)}, configs={flt['n_configs_kept']}, datasets={df['dataset'].nunique()}")

print("\n=== HEADLINE ===")
print(f"conf={df['ext_conf'].mean()*100:.1f}%  any={df['ext_leak'].mean()*100:.1f}%  raw_binary={df['output_leak'].mean()*100:.1f}%")

print("\n=== PER-CATEGORY ===")
for cat in ["Health","Social","Photo/Camera","Finance","Education","Productivity"]:
    cdf = df[df["category"]==cat]
    ni = cdf["full_key"].nunique()
    print(f"  {cat}: n={ni}, raw={cdf['output_leak'].mean()*100:.1f}%, any={cdf['ext_leak'].mean()*100:.1f}%, conf={cdf['ext_conf'].mean()*100:.1f}%")

print("\n=== PER-FAMILY (raw->any->conf) ===")
for fam, attrs in ATTR_FAMILIES.items():
    fdf = df[df["attr"].isin(attrs)]
    print(f"  {fam}: {fdf['output_leak'].mean()*100:.1f}% -> {fdf['ext_leak'].mean()*100:.1f}% -> {fdf['ext_conf'].mean()*100:.1f}%")

print("\n=== PER-CHANNEL ===")
for ch in CHANNELS:
    sub = df[df[f"ch_{ch}_pres"]==1]
    print(f"  {ch}: n_items={sub['full_key'].nunique()}, any={sub[f'ch_{ch}_leak'].mean()*100:.2f}%, conf={sub[f'ch_{ch}_conf'].mean()*100:.2f}%")

print("\n=== RETENTION/INJECTION ===")
for attr in ["identity","gender","location","medical","age","marital status","face","religion","ethnic_clothing"]:
    adf = df[df["attr"]==attr]
    gt1 = adf[adf["input_label"]==1]; gt0 = adf[adf["input_label"]==0]
    ret = gt1["ext_conf"].mean()*100 if len(gt1) else 0
    inj = gt0["ext_conf"].mean()*100 if len(gt0) else 0
    print(f"  {attr}: ret={ret:.1f}%, inj={inj:.1f}%, nconf={int(adf['ext_conf'].sum())}")

print("\n=== MODALITY PAIRS ===")
for mp in ["text→text","image→text","text→image","image→image"]:
    mdf = df[df["modality_pair"]==mp]
    print(f"  {mp}: n={mdf['full_key'].nunique()}, conf={mdf['ext_conf'].mean()*100:.2f}%")

print("\n=== INPUT TYPE ===")
for it in ["text","image","docs"]:
    idf = df[df["in_type"]==it]
    print(f"  {it}: n={idf['full_key'].nunique()}, GT={idf['input_label'].mean()*100:.2f}%, conf={idf['ext_conf'].mean()*100:.2f}%")

print("\n=== PROFILE CONSOLIDATION ===")
ic = df.groupby("full_key")["ext_conf"].sum()
lk = ic[ic>=1]
print(f"  leaking_items={len(lk)}, mean_conf={lk.mean():.2f}, ge2={int((lk>=2).sum())}, ge3={int((lk>=3).sum())}, ge5={int((lk>=5).sum())}")
for cat in ["Health","Social","Photo/Camera","Finance","Education","Productivity"]:
    cdf = df[df["category"]==cat]
    ia = cdf.groupby("full_key")["ext_leak"].sum().mean()
    ic2 = cdf.groupby("full_key")["ext_conf"].sum()
    mc = ic2[ic2>=1].mean() if (ic2>=1).any() else 0
    print(f"  {cat}: any_mean={ia:.2f}, conf_leaking_mean={mc:.2f}")

print("\n=== CHANNEL AGGREGATION LIFT ===")
multi = df[df["n_ext_channels"]>=2]
print(f"  multi-ch items: {multi['full_key'].nunique()}")
agg_conf_total = int(multi["ext_conf"].sum())
any_ch_conf = (multi[[f"ch_{ch}_conf" for ch in CHANNELS]].max(axis=1)==1)
lift = (multi["ext_conf"]==1) & (~any_ch_conf)
lift_pairs = int(lift.sum())
lift_items = multi[lift]["full_key"].nunique()
agg_conf_items = multi[multi["ext_conf"]==1]["full_key"].nunique()
print(f"  agg_conf_pairs={agg_conf_total}, lift_pairs={lift_pairs} ({lift_pairs/agg_conf_total*100:.1f}%)")
print(f"  agg_conf_items={agg_conf_items}, lift_items={lift_items} ({lift_items/agg_conf_items*100:.1f}%)")

print("\n=== DATA SAFETY ===")
CLASS_A = {"location","identity","race","religion","medical"}
CLASS_B = {"age","gender","marital status","disability","nudity"}
CLASS_C = set(df["attr"].unique()) - CLASS_A - CLASS_B
ta = int(df["ext_leak"].sum()); tc = int(df["ext_conf"].sum())
print(f"  total any={ta}, total conf={tc}")
for cn, cs in [("A",CLASS_A),("B",CLASS_B),("C",CLASS_C)]:
    ca = int(df[df["attr"].isin(cs)]["ext_leak"].sum())
    cc = int(df[df["attr"].isin(cs)]["ext_conf"].sum())
    print(f"  Class({cn}): any={ca} ({ca/ta*100:.1f}%), conf={cc} ({cc/tc*100:.1f}%)")
na = int(df[~df["attr"].isin(CLASS_A)]["ext_leak"].sum())
nc2 = int(df[~df["attr"].isin(CLASS_A)]["ext_conf"].sum())
print(f"  Concealed by class-a restriction: any={na/ta*100:.1f}%, conf={nc2/tc*100:.1f}%")
