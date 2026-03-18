# Feature Namespace Convention

## Mandatory Convention: Flat snake_case

All companion code scripts (01–05), Feast feature view definitions, and
production serving code **MUST** use **flat snake_case** feature names
matching the Feast registry entries defined in `02_feature_engineering.py`
and referenced in CODE-02 of the whitepaper.

### Correct (flat snake_case — use this)

```
dl_prb_utilization
ul_throughput_mbps
rrc_setup_success_ratio
ho_intra_freq_success_rate_1h
```

### Incorrect for production (dotted namespace — demo only)

```
ran.kpi.dl_prb_utilisation
ran.kpi.ul_throughput_mbps
ran.kpi.rrc_setup_success_ratio
```

## Why `05_production_patterns.py` Uses Dotted Names

Script 05 uses `ran.kpi.*` dotted namespace **for illustration of the
online feature computation pattern only**. It demonstrates how a serving-time
feature function might namespace features by domain. This is explicitly a
demo convention and **must not** be adopted for production.

## Audit Command

Run this from the companion code root to check for dotted namespace
usage outside of 05:

```bash
grep -rn 'ran\.kpi\.' *.py | grep -v '05_production_patterns.py' | grep -v '#'
```

Any hits indicate a namespace inconsistency that must be resolved before
deployment.

## Spelling

Use US English spelling (`utilization`, not `utilisation`) for all feature
names to match the §3 counter mapping table in the whitepaper.
