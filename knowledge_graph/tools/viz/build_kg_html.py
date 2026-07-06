#!/usr/bin/env python3
"""Build a self-contained interactive Cytoscape.js visualization of the COMPLETE
ChainSight knowledge graph from the current source files.

Unlike a per-file siloed view, this stitches ALL layers into ONE connected graph
using the parameter / metric individuals as the shared spine:

  ontology (TBox)   54 owl:Class  + subClassOf + 17 relationship ObjectProperties
  instance (ABox)   28 TunableParameter + 13 Metric individuals
  causal            param -> process -> metric flow (+/- polarity)
  data-config       26 PostgreSQL config tables (cfg_*)
  data-output       40 PostgreSQL result tables (*_output_*, orchestrator_*, summary_*)
  grounding         ~51 Databricks Unity Catalog FQNs

Cross-layer links:
  param  --sourcedFrom-->   config table  (via cs:source_config)
  config --mapsToClass-->   ontology class (via ontology_class)
  output --producesEntity-> ontology class (via ontology_class)
  class  --groundedIn-->    databricks FQN (via cs:source_table + links.yaml)
  param/metric --instanceOf--> ontology class

Rendering: Cytoscape.js (industry-standard graph library) via CDN, with fcose
clustering layout + dagre left-right layout, per-layer toggles, search, and a
click-to-inspect detail panel. Data is embedded (no file:// fetch / CORS issue).

Usage:
  python knowledge_graph/tools/viz/build_kg_html.py [output.html]
"""
from __future__ import annotations

import json
import os
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

KG_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOP_CLASSES = ["cs:Network", "cs:Product", "cs:Resource", "cs:Inventory",
               "cs:Plan", "cs:Parameter", "cs:Metric"]

# colors -------------------------------------------------------------------- #
CAT_COLOR = {
    "cs:Network": "#38bdf8", "cs:Product": "#2dd4bf", "cs:Resource": "#fb923c",
    "cs:Inventory": "#a78bfa", "cs:Plan": "#34d399", "cs:Parameter": "#f472b6",
    "cs:Metric": "#fbbf24",
}
TYPE_COLOR = {
    "param": "#f472b6", "metric": "#fbbf24", "process": "#fb7185",
    "config": "#60a5fa", "output": "#4ade80", "databricks": "#64748b",
}


# --------------------------------------------------------------------------- #
def _load_json(rel: str) -> dict:
    with open(os.path.join(KG_ROOT, rel), "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(rel: str):
    path = os.path.join(KG_ROOT, rel)
    if yaml is None or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_top(cid: str, classes: dict) -> str | None:
    cur, seen = cid, set()
    while cur and cur not in TOP_CLASSES and cur not in seen:
        seen.add(cur)
        cur = classes.get(cur, {}).get("subClassOf")
    return cur if cur in TOP_CLASSES else None


def build() -> dict:
    nodes: dict[str, dict] = {}   # id -> data
    edges: list[dict] = []

    def add_node(nid, label, ntype, layers, color, color_border=None, **detail):
        if nid in nodes:
            for la in layers:
                if la not in nodes[nid]["layers"]:
                    nodes[nid]["layers"].append(la)
            return nodes[nid]
        nodes[nid] = {
            "id": nid, "label": label, "ntype": ntype,
            "layers": list(layers), "color": color,
            "border": color_border or color,
            "detail": {k: v for k, v in detail.items() if v not in (None, "", [])},
        }
        return nodes[nid]

    def add_edge(src, tgt, rel, layer, color, dashed=False, label=""):
        # endpoints validated at assembly time (nodes may be added in any order)
        edges.append({"source": src, "target": tgt, "rel": rel, "layer": layer,
                      "color": color, "dashed": dashed, "label": label})

    # ---- ontology TBox ---------------------------------------------------- #
    graph = _load_json("schema/chainsight.jsonld").get("@graph", [])
    classes, rel_props = {}, []
    for e in graph:
        t = e.get("@type")
        if t == "owl:Class":
            classes[e["@id"]] = e
        elif t == "owl:ObjectProperty" and e.get("domain") and e.get("range"):
            rel_props.append(e)

    for cid, c in classes.items():
        is_top = cid in TOP_CLASSES
        top = cid if is_top else _resolve_top(cid, classes)
        add_node(
            cid, c.get("label", cid.split(":")[-1]), "class", ["ontology"],
            CAT_COLOR.get(top, "#94a3b8"),
            kind="Top Class" if is_top else "Class",
            id=cid, category=top, subClassOf=c.get("subClassOf"),
            source_type=c.get("cs:source_type"),
            source_config=c.get("cs:source_config"),
            source_table=c.get("cs:source_table"),
            comment=c.get("comment"),
        )
    for cid, c in classes.items():
        if c.get("subClassOf"):
            add_edge(cid, c["subClassOf"], "subClassOf", "ontology", "#475569")
    for p in rel_props:
        add_edge(p["domain"], p["range"], p.get("label", "rel"), "ontology",
                 "#64748b", label=p.get("label", ""))

    # ---- instance ABox: parameters + metrics (the shared spine) ----------- #
    alias = {}  # "param:<alias>" / "metric:<alias>" -> canonical node id

    for ent in _load_json("instances/ontology/parameters.jsonld").get("@graph", []):
        label = ent.get("label", ent["@id"].split(":")[-1])
        canon = "param:" + label
        add_node(canon, label, "param", ["instance"], TYPE_COLOR["param"],
                 kind="Tunable Parameter", id=ent["@id"],
                 param_name=ent.get("cs:param_name"), module=ent.get("cs:module"),
                 source_config=ent.get("cs:source_config"),
                 param_type=ent.get("cs:param_type"),
                 feasibility_horizon=ent.get("cs:feasibility_horizon"),
                 controls=ent.get("cs:controls"), comment=ent.get("cs:comment"))
        for a in {label, ent.get("cs:param_name"), ent["@id"].split("cs:param_")[-1]}:
            if a:
                alias["param:" + a] = canon
        add_edge(canon, "cs:TunableParameter", "instanceOf", "instance",
                 "#334155", dashed=True)
        # param -> config table (cross-layer)
        sc = ent.get("cs:source_config")
        if sc:
            add_edge(canon, "cfg:" + sc, "sourcedFrom", "config",
                     "#3b82f6", dashed=True)

    for ent in _load_json("instances/ontology/metrics.jsonld").get("@graph", []):
        label = ent.get("label", ent["@id"].split(":")[-1])
        canon = "metric:" + label
        add_node(canon, label, "metric", ["instance"], TYPE_COLOR["metric"],
                 kind="Metric", id=ent["@id"], metric_class=ent.get("@type"),
                 metric_name=ent.get("cs:metric_name"),
                 computation_source=ent.get("cs:computation_source"),
                 unit=ent.get("cs:unit"), comment=ent.get("comment"))
        for a in {label, ent.get("cs:metric_name"), ent["@id"].split("cs:metric_")[-1]}:
            if a:
                alias["metric:" + a] = canon
        mclass = ent.get("@type", "cs:Metric")
        add_edge(canon, mclass, "instanceOf", "instance", "#334155", dashed=True)

    # ---- causal layer ----------------------------------------------------- #
    def remap(ref):
        return alias.get(ref, ref)

    dag = _load_yaml("instances/causal/causal_dag.yaml") or {}
    for n in dag.get("nodes", []):
        nid, nt = n["id"], n.get("type", "process")
        if nt in ("parameter", "metric"):
            canon = remap(nid)
            if canon in nodes:
                if "causal" not in nodes[canon]["layers"]:
                    nodes[canon]["layers"].append("causal")
                continue
            # causal-only individual with no ontology twin
            color = TYPE_COLOR["param" if nt == "parameter" else "metric"]
            add_node(canon, n.get("label", nid), "param" if nt == "parameter" else "metric",
                     ["instance", "causal"], color, kind=nt.capitalize(), id=nid,
                     module=n.get("module"), source_config=n.get("source_config"),
                     comment=n.get("description"))
        else:
            add_node(nid, n.get("label", nid), "process", ["causal"],
                     TYPE_COLOR["process"], kind="Process", id=nid,
                     module=n.get("module"), source_config=n.get("source_config"),
                     comment=n.get("description"))
    pol_color = {"+": "#22c55e", "-": "#ef4444"}
    for e in dag.get("edges", []):
        pol = e.get("polarity")
        add_edge(remap(e["source"]), remap(e["target"]),
                 e.get("type", "influences"), "causal",
                 pol_color.get(pol, "#64748b"), label=pol or "")

    # ---- data-config layer ------------------------------------------------ #
    cfg = _load_yaml("instances/data-catalog/config-tables/mapping.yaml") or {}
    for c in cfg.get("configs", []):
        ct = c["config_table"]
        cid = "cfg:" + ct
        add_node(cid, ct, "config", ["config"], TYPE_COLOR["config"],
                 kind="Config Table", id=ct, db_table=c.get("db_table"),
                 ontology_class=c.get("ontology_class"),
                 required=c.get("required"),
                 columns=", ".join(str(x) for x in c.get("schema", {}).get("columns", [])))
        oc = c.get("ontology_class")
        if oc:
            add_edge(cid, oc, "mapsToClass", "config", "#3b82f6")

    # ---- data-output layer ------------------------------------------------ #
    out = _load_yaml("instances/data-catalog/output-tables/mapping.yaml") or {}
    for o in out.get("outputs", []):
        oid = "out:" + o["db_table"]
        add_node(oid, o["db_table"], "output", ["output"], TYPE_COLOR["output"],
                 kind="Output Table", id=o["db_table"], module=o.get("module"),
                 category=o.get("category"), ontology_class=o.get("ontology_class"),
                 source_config=o.get("source_config"), source_df=o.get("source_df"),
                 columns=", ".join(str(x) for x in (o.get("key_columns") or [])),
                 comment=o.get("description"))
        oc = o.get("ontology_class")
        if oc:
            add_edge(oid, oc, "producesEntity", "output", "#22c55e")

    # ---- grounding layer -------------------------------------------------- #
    reg = _load_yaml("instances/data-catalog/registry.yaml") or {}
    for fqn in reg.get("tables", []):
        add_node("db:" + fqn, fqn.split(".")[-1], "databricks", ["grounding"],
                 TYPE_COLOR["databricks"], kind="Databricks Table", id=fqn,
                 service=reg.get("service"))
    # class -> databricks (from cs:source_table)
    for cid, c in classes.items():
        for fqn in (c.get("cs:source_table") or []):
            node_id = "db:" + fqn
            if node_id not in nodes:
                add_node(node_id, fqn.split(".")[-1], "databricks", ["grounding"],
                         TYPE_COLOR["databricks"], kind="Databricks Table", id=fqn)
            add_edge(cid, node_id, "groundedIn", "grounding", "#64748b", dashed=True)
    # databricks -> class (from links.yaml episodic links)
    links = _load_yaml("instances/data-catalog/links.yaml") or {}
    for lk in links.get("links", []):
        node_id = "db:" + lk["fqn"]
        if node_id not in nodes:
            add_node(node_id, lk["fqn"].split(".")[-1], "databricks", ["grounding"],
                     TYPE_COLOR["databricks"], kind="Databricks Table", id=lk["fqn"])
        for ref in lk.get("ontology_refs", []):
            add_edge(node_id, ref, "referencedBy", "grounding", "#475569", dashed=True)

    # ---- assemble cytoscape elements -------------------------------------- #
    elements = []
    for n in nodes.values():
        elements.append({"data": {
            "id": n["id"], "label": n["label"], "ntype": n["ntype"],
            "layers": n["layers"], "color": n["color"], "border": n["border"],
            "detail": n["detail"],
        }})
    for i, e in enumerate(e for e in edges if e["source"] in nodes and e["target"] in nodes):
        elements.append({"data": {
            "id": "e%d" % i, "source": e["source"], "target": e["target"],
            "rel": e["rel"], "layer": e["layer"], "color": e["color"],
            "dashed": "dash" if e["dashed"] else "solid", "label": e["label"],
        }})

    stats = {
        "class": sum(1 for n in nodes.values() if n["ntype"] == "class"),
        "param": sum(1 for n in nodes.values() if n["ntype"] == "param"),
        "metric": sum(1 for n in nodes.values() if n["ntype"] == "metric"),
        "process": sum(1 for n in nodes.values() if n["ntype"] == "process"),
        "config": sum(1 for n in nodes.values() if n["ntype"] == "config"),
        "output": sum(1 for n in nodes.values() if n["ntype"] == "output"),
        "databricks": sum(1 for n in nodes.values() if n["ntype"] == "databricks"),
        "nodes": len(nodes),
        "edges": sum(1 for e in edges if e["source"] in nodes and e["target"] in nodes),
    }
    return {"elements": elements, "stats": stats}


# --------------------------------------------------------------------------- #
HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ChainSight Knowledge Graph</title>
<script src="https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/layout-base@2.0.1/layout-base.js"></script>
<script src="https://unpkg.com/cose-base@2.2.0/cose-base.js"></script>
<script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
<script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
<script src="https://unpkg.com/cytoscape-dagre@2.5.0/cytoscape-dagre.js"></script>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  html,body{height:100%;font-family:'Segoe UI','Inter',system-ui,sans-serif;background:#0b0e17;color:#cbd5e1;overflow:hidden}
  #bar{height:50px;display:flex;align-items:center;gap:14px;padding:0 18px;background:#0f1320;border-bottom:1px solid #1e293b;position:relative;z-index:5}
  #bar h1{font-size:15px;font-weight:700;letter-spacing:.4px;color:#e2e8f0;white-space:nowrap}
  #bar h1 span{background:linear-gradient(135deg,#38bdf8,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
  #search{background:#111827;border:1px solid #1e293b;border-radius:7px;color:#e2e8f0;font-size:13px;padding:7px 12px;width:220px;outline:none}
  #search:focus{border-color:#38bdf8}
  .lbtn{padding:6px 12px;border:1px solid #1e293b;border-radius:7px;background:#111827;color:#94a3b8;font-size:12.5px;font-weight:600;cursor:pointer}
  .lbtn:hover{color:#e2e8f0;border-color:#334155}
  .lbtn.active{color:#0b0e17;background:linear-gradient(135deg,#38bdf8,#a78bfa);border-color:transparent}
  #stats{margin-left:auto;font-size:12px;color:#64748b;white-space:nowrap}
  #cy{position:absolute;top:50px;left:230px;right:300px;bottom:0}
  #left{position:absolute;top:50px;left:0;bottom:0;width:230px;background:#0f1320;border-right:1px solid #1e293b;padding:14px;overflow:auto}
  #left h3{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin:14px 0 8px}
  #left h3:first-child{margin-top:0}
  .layer{display:flex;align-items:center;gap:8px;font-size:12.5px;margin:6px 0;cursor:pointer;user-select:none}
  .layer input{accent-color:#38bdf8;width:14px;height:14px;cursor:pointer}
  .layer .sw{width:11px;height:11px;border-radius:3px;flex:0 0 auto}
  .lg{display:flex;align-items:center;gap:8px;font-size:11.5px;margin:5px 0;color:#94a3b8}
  .dot{width:11px;height:11px;border-radius:50%;flex:0 0 auto}
  .ln{width:18px;height:0;border-top:2px solid;flex:0 0 auto}
  #side{position:absolute;top:50px;right:0;bottom:0;width:300px;background:#0f1320;border-left:1px solid #1e293b;padding:16px;overflow:auto}
  #side h3{font-size:11px;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin-bottom:10px}
  #side .empty{color:#475569;font-size:13px;font-style:italic}
  .badge{display:inline-block;padding:3px 11px;border-radius:20px;font-size:11px;font-weight:700;color:#0b0e17;margin-bottom:12px}
  .kv{margin:9px 0;font-size:12.5px;line-height:1.5}
  .kv .k{color:#64748b;display:block;font-size:10.5px;text-transform:uppercase;letter-spacing:.5px}
  .kv .v{color:#e2e8f0;word-break:break-word}
</style>
</head>
<body>
<div id="bar">
  <h1><span>ChainSight</span> KG</h1>
  <input id="search" placeholder="搜索节点…"/>
  <button class="lbtn" id="lay-fcose">聚类布局</button>
  <button class="lbtn" id="lay-dagre">因果流向 (LR)</button>
  <button class="lbtn" id="fit">适应屏幕</button>
  <div id="stats"></div>
</div>
<div id="left">
  <h3>图层 Layers</h3>
  <div id="layers"></div>
  <h3>节点 Nodes</h3>
  <div id="lg-nodes"></div>
  <h3>连线 Edges</h3>
  <div id="lg-edges"></div>
</div>
<div id="cy"></div>
<div id="side"><h3>详情 Details</h3><div class="empty">点击任意节点查看详情</div></div>
<script>
const DATA = __DATA__;
const STATS = __STATS__;

const LAYER_DEF = [
  ['ontology','本体 (类/关系)','#38bdf8'],
  ['instance','实例 (参数/指标)','#f472b6'],
  ['causal','因果链 (过程/影响)','#fb7185'],
  ['config','配置表','#60a5fa'],
  ['output','输出表','#4ade80'],
  ['grounding','Databricks 落地','#64748b'],
];
const NODE_LG = [
  ['#38bdf8','Network 类'],['#2dd4bf','Product 类'],['#fb923c','Resource 类'],
  ['#a78bfa','Inventory 类'],['#34d399','Plan 类'],['#f472b6','参数 / Parameter'],
  ['#fbbf24','指标 / Metric'],['#fb7185','过程 / Process'],
  ['#60a5fa','配置表'],['#4ade80','输出表'],['#64748b','Databricks 表'],
];
const EDGE_LG = [
  ['#475569','subClassOf'],['#64748b','关系 (domain→range)'],
  ['#22c55e','影响 +'],['#ef4444','影响 −'],
  ['#3b82f6','参数→配置 / 配置→类'],['#4ade80','输出→类'],['#64748b','落地 groundedIn'],
];

const active = new Set(LAYER_DEF.map(l => l[0]));

cytoscape.warnings(false);
const cy = cytoscape({
  container: document.getElementById('cy'),
  elements: DATA.elements,
  wheelSensitivity: 0.25,
  style: [
    {selector:'node', style:{
      'label':'data(label)','font-size':7,'color':'#e2e8f0','text-valign':'center',
      'text-halign':'center','text-outline-width':1.4,'text-outline-color':'#0b0e17',
      'background-color':'data(color)','border-color':'data(border)','border-width':1,
      'width':16,'height':16,'text-max-width':90,'text-wrap':'ellipsis',
    }},
    {selector:'node[ntype="class"]', style:{'shape':'round-rectangle','width':26,'height':20,'font-size':8}},
    {selector:'node[ntype="param"]', style:{'shape':'ellipse','width':22,'height':22}},
    {selector:'node[ntype="metric"]', style:{'shape':'diamond','width':24,'height':24}},
    {selector:'node[ntype="process"]', style:{'shape':'hexagon','width':18,'height':18}},
    {selector:'node[ntype="config"]', style:{'shape':'round-tag','width':22,'height':16}},
    {selector:'node[ntype="output"]', style:{'shape':'tag','width':22,'height':16}},
    {selector:'node[ntype="databricks"]', style:{'shape':'barrel','width':20,'height':16,'font-size':6}},
    {selector:'edge', style:{
      'width':1.1,'line-color':'data(color)','target-arrow-color':'data(color)',
      'target-arrow-shape':'triangle','arrow-scale':0.7,'curve-style':'bezier',
      'opacity':0.7,'font-size':6,'color':'#94a3b8','text-outline-width':2,
      'text-outline-color':'#0b0e17','label':'data(label)',
    }},
    {selector:'edge[dashed="dash"]', style:{'line-style':'dashed'}},
    {selector:'.dim', style:{'opacity':0.07,'text-opacity':0}},
    {selector:'.hit', style:{'border-width':3,'border-color':'#fff'}},
    {selector:'node:selected', style:{'border-width':3,'border-color':'#fff'}},
  ],
  layout:{name:'fcose', quality:'default', animate:false, randomize:true,
          nodeRepulsion:9000, idealEdgeLength:70, nodeSeparation:80, packComponents:true},
});

function applyLayers(){
  cy.batch(() => {
    cy.nodes().forEach(n => {
      const vis = (n.data('layers')||[]).some(l => active.has(l));
      n.style('display', vis ? 'element' : 'none');
    });
    cy.edges().forEach(e => {
      const on = active.has(e.data('layer'));
      const ends = e.source().style('display')!=='none' && e.target().style('display')!=='none';
      e.style('display', (on && ends) ? 'element' : 'none');
    });
  });
}

function runLayout(name){
  const opts = name==='dagre'
    ? {name:'dagre', rankDir:'LR', nodeSep:18, rankSep:90, animate:false,
       fit:true, eles: cy.elements(':visible')}
    : {name:'fcose', animate:false, randomize:true, nodeRepulsion:9000,
       idealEdgeLength:70, nodeSeparation:80, packComponents:true, eles: cy.elements(':visible')};
  cy.layout(opts).run();
  document.getElementById('lay-fcose').classList.toggle('active', name!=='dagre');
  document.getElementById('lay-dagre').classList.toggle('active', name==='dagre');
}

// ---- side detail ---------------------------------------------------------- #
const FIELD_ORDER = ['kind','id','category','instanceOf','subClassOf','metric_class',
  'module','source_type','source_config','source_df','db_table','ontology_class',
  'param_type','feasibility_horizon','unit','metric_name','param_name',
  'computation_source','controls','required','columns','service',
  'source_table','comment'];
function showDetail(d){
  let h = '<h3>详情 Details</h3>';
  const col = (d._color)||'#94a3b8';
  h += '<span class="badge" style="background:'+col+'">'+(d.kind||'')+'</span>';
  const seen = new Set();
  FIELD_ORDER.forEach(k => {
    if(seen.has(k)) return; seen.add(k);
    let v = d[k];
    if(v===undefined||v===null||v==='') return;
    if(Array.isArray(v)) v = v.join('<br>');
    h += '<div class="kv"><span class="k">'+k.replace(/_/g,' ')+'</span><span class="v">'+v+'</span></div>';
  });
  document.getElementById('side').innerHTML = h;
}
cy.on('tap','node', evt => {
  const n = evt.target;
  const d = Object.assign({}, n.data('detail'));
  d._color = n.data('color');
  showDetail(d);
});

// ---- search --------------------------------------------------------------- #
document.getElementById('search').addEventListener('input', e => {
  const q = e.target.value.trim().toLowerCase();
  cy.elements().removeClass('dim hit');
  if(!q) return;
  const hit = cy.nodes().filter(n => (n.data('label')||'').toLowerCase().includes(q)
                                  || ((n.data('detail')||{}).id||'').toLowerCase().includes(q));
  if(hit.length){
    cy.elements().addClass('dim');
    hit.removeClass('dim').addClass('hit');
    hit.neighborhood().removeClass('dim');
  }
});

// ---- left panel ----------------------------------------------------------- #
function buildLeft(){
  document.getElementById('layers').innerHTML = LAYER_DEF.map(([k,label,c]) =>
    '<label class="layer"><input type="checkbox" data-layer="'+k+'" checked>'
    + '<span class="sw" style="background:'+c+'"></span>'+label+'</label>').join('');
  document.querySelectorAll('#layers input').forEach(cb =>
    cb.addEventListener('change', () => {
      cb.checked ? active.add(cb.dataset.layer) : active.delete(cb.dataset.layer);
      applyLayers();
    }));
  document.getElementById('lg-nodes').innerHTML = NODE_LG.map(([c,l]) =>
    '<div class="lg"><span class="dot" style="background:'+c+'"></span>'+l+'</div>').join('');
  document.getElementById('lg-edges').innerHTML = EDGE_LG.map(([c,l]) =>
    '<div class="lg"><span class="ln" style="border-color:'+c+'"></span>'+l+'</div>').join('');
}

document.getElementById('lay-fcose').onclick = () => runLayout('fcose');
document.getElementById('lay-dagre').onclick = () => runLayout('dagre');
document.getElementById('fit').onclick = () => cy.fit(cy.elements(':visible'), 40);

document.getElementById('stats').textContent =
  STATS.class+' 类 · '+STATS.param+' 参数 · '+STATS.metric+' 指标 · '+STATS.process
  +' 过程 · '+STATS.config+' 配置 · '+STATS.output+' 输出 · '+STATS.databricks
  +' DB | '+STATS.nodes+' 节点 / '+STATS.edges+' 边';

buildLeft();
document.getElementById('lay-fcose').classList.add('active');
cy.ready(() => cy.fit(cy.elements(':visible'), 40));
</script>
</body>
</html>
"""


def main() -> None:
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "kg_graph.html")
    g = build()
    html = (HTML
            .replace("__DATA__", json.dumps({"elements": g["elements"]}, ensure_ascii=False))
            .replace("__STATS__", json.dumps(g["stats"], ensure_ascii=False)))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print("stats:", g["stats"])
    print("wrote:", out)


if __name__ == "__main__":
    main()
