"""Readable notebook results and an explicit, small results-only download.

Can also be downloaded as a standalone script to summarize an existing run.
No checkpoints, merged models, data dumps, or existing ZIPs enter the bundle.
"""
import csv
import io
import json
from pathlib import Path
import zipfile
import numpy as np

NAMES={"popularity":"Popularity", "supervised_policy":"Before GRPO (SFT)",
       "grpo_policy":"After GRPO", "direct_value":"Direct value ranking",
       "greedy_list_value":"Greedy list optimizer"}
ORDER=["popularity","supervised_policy","grpo_policy","direct_value","greedy_list_value"]


def escaped(value):
    return str(value).replace("|","\\|").replace("\n"," ").replace("<","&lt;").replace(">","&gt;")


def average(values):return float(np.mean(values)) if len(values) else None


def summarize(root,run_dir=None):
    root=Path(root)
    if run_dir is not None:
        folder=Path(run_dir)
        if not folder.is_absolute():folder=root/folder
        path=folder/"run.json"
    else:
        path=root/"results/llm/run.json"
        if not path.exists():raise FileNotFoundError(
            "No completed LLM run at results/llm/run.json. Finish the training cell "
            "or pass run_dir='results/gpu-check' to inspect that short check. "
            "The bundled CPU example is not your LLM result.")
    run=json.loads(path.read_text());evaluation=run.get("evaluation",[])
    if not evaluation:raise ValueError("Run has no evaluation contexts")
    methods=[m for m in ORDER if all(m in e["methods"] for e in evaluation)]
    simulated=all("simulated_value" in e["methods"][m] for e in evaluation for m in methods)
    value_key="simulated_value" if simulated else "proxy_value"
    metrics=[]
    values={m:np.array([e["methods"][m][value_key] for e in evaluation],dtype=float) for m in methods}
    base=average(values.get("supervised_policy",[]))
    for m in methods:
        records=[e["methods"][m] for e in evaluation];value=float(values[m].mean())
        valid=[not x.get("fallback",False) for x in records]
        metrics.append(dict(method=m,label=NAMES[m],value=value,
            change_vs_sft_pct=None if base in [None,0] else 100*(value/base-1),
            valid_slates=sum(valid),contexts=len(evaluation),valid_rate=sum(valid)/len(valid),
            fallback_rate=1-sum(valid)/len(valid),
            valid_only_value=average([x[value_key] for x,v in zip(records,valid) if v])))
    comparisons={}
    if "grpo_policy" in values:
        rng=np.random.default_rng(99)
        for other in ["supervised_policy","greedy_list_value","direct_value"]:
            if other not in values:continue
            d=values["grpo_policy"]-values[other]
            # Bound intermediate memory for large runs.
            boots=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(1000)])
            comparisons[other]=dict(mean_delta=float(d.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())
    grpo=next((x for x in metrics if x["method"]=="grpo_policy"),None)
    if grpo and grpo["valid_slates"]==0:
        verdict="GRPO produced no valid lists. The displayed value comes entirely from the fallback optimizer."
    elif grpo and grpo["fallback_rate"]>0:
        verdict=f"GRPO needs fallback on {grpo['fallback_rate']:.0%} of contexts. Its reported value mixes policy and fallback results."
    elif "greedy_list_value" in comparisons:
        comp=comparisons["greedy_list_value"];lo,hi=comp["ci95"]
        if lo<=0<=hi:verdict="This evaluation does not clearly separate GRPO from the greedy list optimizer."
        elif lo>0:verdict="GRPO scores higher than the greedy optimizer in this evaluation."
        else:verdict="The greedy list optimizer scores higher than GRPO in this evaluation."
    else:verdict="Compare GRPO with SFT and the greedy baseline before choosing a policy."
    contexts={}
    for candidate in [root/"data/test.jsonl",root/"results/cpu/examples.jsonl"]:
        if candidate.exists():
            for line in candidate.read_text().splitlines():
                if line.strip():
                    row=json.loads(line);contexts[row["id"]]=row
    examples=[]
    for entry in evaluation[:10]:
        row=contexts.get(entry["id"],{})
        queries=entry.get("candidate_queries") or [c["query"] for c in row.get("candidates",[])]
        def names(ids):
            return [queries[i] if isinstance(i,int) and 0<=i<len(queries) else f"Candidate #{i}" for i in ids]
        examples.append(dict(id=entry["id"],prefix=entry.get("prefix",row.get("prefix","unavailable")),
            session=entry.get("session",row.get("session","")),methods={m:dict(
                raw=names(entry["methods"][m].get("raw",[])),
                shown=names(entry["methods"][m].get("deployed",[])),
                fallback=entry["methods"][m].get("fallback",False)) for m in methods}))
    return dict(title="Autocomplete experiment results",source=str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
        kind="Synthetic expected value" if simulated else "Reward-model estimate",
        is_tiny=bool(run.get("config",{}).get("tiny")),contexts=len(evaluation),verdict=verdict,
        model=run.get("config",{}).get("model","unknown"),steps=run.get("config",{}).get("steps"),
        sft_steps=run.get("config",{}).get("sft_steps"),prompt_tokens=run.get("prompt_tokens"),
        metrics=metrics,comparisons=comparisons,examples=examples,history=run.get("history",[]))


def markdown(report):
    lines=["# Autocomplete experiment results", "", "**"+report["verdict"]+"**", "",
        f"{report['contexts']} evaluation contexts · {escaped(report['model'])} · {report['sft_steps']} SFT / {report['steps']} GRPO steps", "",
        "**"+report["kind"]+"; no measured customer GMV.** Higher value and higher valid-list rate are better.", ""]
    if report["is_tiny"]:lines += ["**Random tiny-model integration check only; this is not a pretrained-model quality result.**", ""]
    lines += ["| Method | Value ↑ | Change vs SFT | Valid lists ↑ | Fallbacks ↓ |",
              "|---|---:|---:|---:|---:|"]
    for m in report["metrics"]:
        change="—" if m["change_vs_sft_pct"] is None else f"{m['change_vs_sft_pct']:+.1f}%"
        lines.append(f"| {m['label']} | {m['value']:.3f} | {change} | {m['valid_slates']}/{m['contexts']} ({m['valid_rate']:.0%}) | {m['fallback_rate']:.0%} |")
    lines += ["", "Value is measured after fallback. A high value with frequent fallback is not evidence of a good learned policy.", ""]
    for baseline,c in report["comparisons"].items():
        lines.append(f"GRPO minus {NAMES[baseline]}: **{c['mean_delta']:+.3f}**; paired context-bootstrap 95% interval [{c['ci95'][0]:+.3f}, {c['ci95'][1]:+.3f}].")
    lines += ["", "These intervals describe variation across this evaluation's contexts, not training-seed uncertainty or causal GMV lift.", "",
        "## What do the suggestions look like?", ""]
    for e in report["examples"][:3]:
        lines += [f"### Typed: {escaped(e['prefix'])}", "",escaped(e["session"]), "",
            "| Position | Before GRPO | GRPO raw output | Actually shown | Greedy baseline |", "|---|---|---|---|---|"]
        for i in range(5):
            def val(method,key):
                xs=e["methods"].get(method,{}).get(key,[])
                return escaped(xs[i]) if i<len(xs) else "—"
            lines.append(f"| {i+1} | {val('supervised_policy','raw')} | {val('grpo_policy','raw')} | {val('grpo_policy','shown')} | {val('greedy_list_value','shown')} |")
        lines += ["", "Fallback used: **"+("yes" if e["methods"].get("grpo_policy",{}).get("fallback") else "no")+"**.", ""]
    lines += ["## Run details", "",f"Input tokens: `{report['prompt_tokens']}`. Output budget: five candidate tokens.",
        "SGLang latency is not inferred from training time; run the separate latency benchmark.", "",
        "Small download contents: this summary, metrics CSV, readable examples, and compact evaluation details. No model weights or merged models."]
    return "\n".join(lines)+"\n"


def write_small_report(root,report):
    root=Path(root);destination=root/"results/readable";destination.mkdir(parents=True,exist_ok=True)
    summary=markdown(report)
    output=io.StringIO();fields=list(report["metrics"][0])
    writer=csv.DictWriter(output,fieldnames=fields);writer.writeheader();writer.writerows(report["metrics"])
    # Explicit whitelist. Never archive the entire results directory.
    contents={"summary.md":summary,"metrics.csv":output.getvalue(),
        "examples.json":json.dumps(report["examples"],indent=2,ensure_ascii=False),
        "details.json":json.dumps({k:v for k,v in report.items() if k not in ["examples","history"]},indent=2)}
    for name,content in contents.items():(destination/name).write_text(content)
    bundle=destination/"autocomplete-results-small.zip"
    with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
        for name,content in contents.items():z.writestr(name,content)
    return bundle


def show_results(root=None,run_dir=None,download=False):
    root=Path.cwd() if root is None else Path(root)
    report=summarize(root,run_dir);bundle=write_small_report(root,report)
    try:
        from IPython.display import display,Markdown
        display(Markdown(markdown(report)))
    except ImportError:print(markdown(report))
    print(f"Small results download: {bundle.stat().st_size/1024:.1f} KB — {bundle.name}")
    if download:
        try:
            from google.colab import files
            files.download(str(bundle))
        except ImportError:print(bundle)
    return report,bundle


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default=".");p.add_argument("--run-dir")
    args=p.parse_args();show_results(args.root,args.run_dir)
