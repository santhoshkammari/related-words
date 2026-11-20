import sys, json, collections
pos=collections.Counter(); keys=collections.Counter(); skeys=collections.Counter(); n=0; ex=0; sens=0; shown=0; bad=0
for line in sys.stdin:
    try: d=json.loads(line)
    except Exception: bad+=1; continue
    n+=1; pos[d.get("pos")]+=1; keys.update(d.keys())
    for s in d.get("senses",[]):
        sens+=1; skeys.update(s.keys())
        if s.get("examples"): ex+=1
    if shown<2 and len(d.get("senses",[]))>2 and d.get("pos")=="verb" and len(d["word"])>5:
        shown+=1; print(d["word"], d["pos"], json.dumps(d["senses"][0])[:600])
print("entries",n,"bad",bad,"senses",sens,"senses_with_examples",ex)
print("pos",pos.most_common(8)); print("entry keys",[k for k,_ in keys.most_common(14)]); print("sense keys",[k for k,_ in skeys.most_common(14)])
