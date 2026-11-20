import os
RUN=os.environ.get("RUN_PY","skills/related-words/run.py")  # path to run.py
import sys, importlib.util, numpy as np, pandas as pd, torch
sys.argv=["x","--serve-never"]
src=open(RUN).read().rsplit("\nmain()",1)[0]
ns={"__file__":RUN}; exec(src,ns)
new=ns["load"](); old=dict(new); R="/tmp/vs_old/"
mo=pd.read_parquet(R+"meta.parquet"); old.update(E=torch.from_numpy(np.load(R+"emb256.npy").astype(np.float32)),lem=mo.lemma.values,pos=mo.pos.values,zipf=mo.zipf.values,gloss=mo.gloss_leaf.values)
T=["The committee's ambiguous remarks sparked a protracted debate, and critics argued the proposal was unrealistic.",
"She felt a mixture of nostalgia and dread as she walked through the abandoned house where she grew up.",
"The economy is slowing because inflation keeps rising and consumers are cutting back on spending.",
"He was known for his meticulous attention to detail and relentless work ethic.",
"The politician dodged the question with vague, evasive answers.",
"A sudden storm forced the hikers to abandon their trek and shelter in a cave.",
"The company's aggressive expansion left it dangerously overextended and unable to pay its debts.",
"After years of neglect the garden had become a tangled wilderness of weeds and brambles.",
"The scientist was reluctant to publish results she considered preliminary and inconclusive.",
"Their friendship survived despite constant bickering over trivial matters."]
ov=[];ov1=[]
for t in T:
    a=[w["word"] for w in ns["suggest"](old,t,5)["words"][:10]]; b=[w["word"] for w in ns["suggest"](new,t,5)["words"][:10]]
    ov.append(len(set(a)&set(b))/max(len(a),1)); ov1.append(a[0]==b[0]); print(t[:40],"|",a[:4],"->",b[:4])
print("top10 passage-list overlap",np.mean(ov).round(2),"top1 same",np.mean(ov1))
