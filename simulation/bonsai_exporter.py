"""
bonsai_exporter.py — converts a 2D sim CircuitModel into LBP.Torch
Bonsai XML that can be pasted directly into a Bonsai workflow.

Structure of the generated XML
───────────────────────────────
1. Input preparation branch
   For each active input (sensor or ConstantLayer):
     SubscribeSubject("{name}Input") → ToTensor
   All ToTensor outputs → CombineLatest
   Timer → WithLatestFrom(CombineLatest) → Item2 → BehaviorSubject("NetworkInput")

2. Graph construction (List<GraphNode> chain)
   For each active sensor:
     InputLayer(i) → [Dynamics if tau] → [activation if not linear]
   Then Linear(W_sensor_to_layer) per connection.
   For each active ConstantLayer:
     InputLayer(j) → Linear(W_const_to_layer)
   Fan-in with Zip → Join → Add when multiple sources feed one layer.
   Layer dynamics (LeakyLayer → Dynamics) and activation inside the chain.
   → CreateTorchModel → BehaviorSubject("ModelCreated")

3. Forward pass
   SubscribeSubject("NetworkInput") → TorchModelForward → BehaviorSubject("Output")
   → SubscribeWhen(SubscribeSubject("ModelCreated"))

Rules
─────
  • Only sensors/layers that have a path to the motor layer are included.
  • Sensor dynamics and activation live inside the model graph, not in the
    input prep branch.
  • Each sensor keeps its own InputLayer (no block-matrix merging).
"""

import numpy as np
from collections import defaultdict


def generate_bonsai_xml(circuit, model_name='SimulatorModel'):
    """
    Convert a CircuitModel to Bonsai WorkflowBuilder XML.

    Parameters
    ----------
    circuit    : CircuitModel with .sensors, .layers, .connections
    model_name : name used in CreateTorchModel / TorchModelForward

    Returns
    -------
    str — Bonsai XML ready for clipboard paste into a Bonsai workflow.
    """
    from neurons import LeakyLayer, ConstantLayer, MatsuokaLayer, AdaptiveLayer

    sensors = circuit.sensors
    layers  = circuit.layers
    conns   = circuit.connections

    # ── Accumulators ──────────────────────────────────────────────────────────

    nodes_xml, edges_xml = [], []
    _n = [0]

    def node(xml_str):
        nodes_xml.append(xml_str)
        idx = _n[0]; _n[0] += 1
        return idx

    def edge(frm, to, label='Source1'):
        edges_xml.append(f'      <Edge From="{frm}" To="{to}" Label="{label}" />')

    def fmt_W(W):
        W = np.atleast_2d(np.asarray(W, dtype=float))
        rows = [', '.join(f'{v:.6g}' for v in row) for row in W]
        return '[' + '; '.join(rows) + ']'

    # ── Topological sort ──────────────────────────────────────────────────────

    sensor_map = {s.name: s for s in sensors}
    layer_map  = {l.name: l for l in layers}

    by_tgt = defaultdict(list)
    for conn in conns:
        by_tgt[conn.tgt].append((conn.src, np.asarray(conn.W, dtype=float)))

    deps = {l.name: set() for l in layers}
    for conn in conns:
        if conn.src in layer_map and conn.tgt in deps:
            deps[conn.tgt].add(conn.src)

    ordered, ready, remaining = [], set(sensor_map), list(layers)
    while remaining:
        progress = False
        for lyr in list(remaining):
            if deps[lyr.name].issubset(ready):
                ordered.append(lyr); ready.add(lyr.name)
                remaining.remove(lyr); progress = True
        if not progress:
            ordered.extend(remaining); break

    const_layers  = [l for l in ordered if isinstance(l, ConstantLayer)]
    active_layers = [l for l in ordered if not isinstance(l, ConstantLayer)]

    # ── Rule: only include nodes that influence motor output ──────────────────

    connected_srcs = {src for src, _, _ in conns}
    active_sensors = [s for s in sensors      if s.name in connected_srcs]
    active_consts  = [l for l in const_layers if l.name in connected_srcs]

    warnings  = []
    chain_end = {}       # name → last graph-construction node index
    main_end  = None     # tail of the graph-construction chain

    # All active inputs in order (sensors first, then constants)
    all_inputs = [(s.name, i) for i, s in enumerate(active_sensors)]
    all_inputs += [(l.name, len(active_sensors) + j) for j, l in enumerate(active_consts)]

    _ACT_MAP = {
        'relu': 'ReLU', 'leakyrelu': 'LeakyReLU',
        'elu': 'ELU', 'sigmoid': 'Sigmoid',
        'tanh': 'Tanh', 'linear': 'Linear',
    }

    def cap(n):
        """Capitalise only the first letter; leave the rest untouched."""
        return n[:1].upper() + n[1:] if n else n

    # ══════════════════════════════════════════════════════════════════════════
    # 1. INPUT PREPARATION BRANCH
    #    Sensors:   SubscribeSubject → ToTensor
    #    Constants: CreateTensor([value]) — value known at export time
    #    All tensor ends → CombineLatest
    #    Timer → WithLatestFrom(CombineLatest) → Item2 → NetworkInput
    # ══════════════════════════════════════════════════════════════════════════

    const_map = {l.name: l for l in active_consts}
    to_tensor_ends = []   # node index of the last tensor node for each input (in order)

    for name, _ in all_inputs:
        if name in const_map:
            # ConstantLayer — emit a fixed tensor directly
            lyr = const_map[name]
            val = lyr._value
            val_list = (val * np.ones(lyr.n))[:lyr.n].tolist() if val.size == 1 else val[:lyr.n].tolist()
            val_str  = '[' + ', '.join(f'{v:g}' for v in val_list) + ']'
            ct_i = node(
                f'      <Expression xsi:type="Combinator">\n'
                f'        <Combinator xsi:type="p1:CreateTensor">\n'
                f'          <p1:Value>{val_str}</p1:Value>\n'
                f'          <p1:DataType>Float32</p1:DataType>\n'
                f'        </Combinator>\n'
                f'      </Expression>'
            )
            to_tensor_ends.append(ct_i)
        else:
            # Sensor — subscribe to external data source
            s = next((s for s in active_sensors if s.name == name), None)
            subj = (getattr(s, 'robot_address', '') or '').strip() or f'{cap(name)}Input'
            sub_i = node(
                f'      <Expression xsi:type="SubscribeSubject">\n'
                f'        <Name>{subj}</Name>\n'
                f'      </Expression>'
            )
            tt_i = node(
                f'      <Expression xsi:type="Combinator">\n'
                f'        <Combinator xsi:type="p1:ToTensor">\n'
                f'          <p1:Copy>true</p1:Copy>\n'
                f'          <p1:DType>Float32</p1:DType>\n'
                f'          <p1:Scale>1</p1:Scale>\n'
                f'          <p1:Shift>0</p1:Shift>\n'
                f'        </Combinator>\n'
                f'      </Expression>'
            )
            edge(sub_i, tt_i)
            to_tensor_ends.append(tt_i)

    if len(to_tensor_ends) == 0:
        warnings.append('No active inputs — input prep branch not generated.')
        cl_out = None
    elif len(to_tensor_ends) == 1:
        cl_out = to_tensor_ends[0]   # single input, no CombineLatest needed
    else:
        cl_i = node(
            '      <Expression xsi:type="Combinator">\n'
            '        <Combinator xsi:type="rx:CombineLatest" />\n'
            '      </Expression>'
        )
        for k, tt in enumerate(to_tensor_ends):
            edge(tt, cl_i, f'Source{k+1}')
        cl_out = cl_i

    if cl_out is not None:
        timer_i = node(
            '      <Expression xsi:type="Combinator">\n'
            '        <Combinator xsi:type="rx:Timer">\n'
            '          <rx:DueTime>PT0S</rx:DueTime>\n'
            '          <rx:Period>PT0.02S</rx:Period>\n'
            '        </Combinator>\n'
            '      </Expression>'
        )
        wlf_i = node(
            '      <Expression xsi:type="Combinator">\n'
            '        <Combinator xsi:type="rx:WithLatestFrom" />\n'
            '      </Expression>'
        )
        edge(timer_i, wlf_i, 'Source1')
        edge(cl_out,  wlf_i, 'Source2')

        if len(all_inputs) > 1:
            # Item2 extracts the sensor/drive tuple from Tuple<long, Tuple<...>>
            item2_i = node(
                '      <Expression xsi:type="MemberSelector">\n'
                '        <Selector>Item2</Selector>\n'
                '      </Expression>'
            )
            edge(wlf_i, item2_i)
            net_src = item2_i
        else:
            # Single input: WithLatestFrom gives Tuple<long, Tensor>; Item2 = Tensor
            item2_i = node(
                '      <Expression xsi:type="MemberSelector">\n'
                '        <Selector>Item2</Selector>\n'
                '      </Expression>'
            )
            edge(wlf_i, item2_i)
            net_src = item2_i

        net_input_i = node(
            '      <Expression xsi:type="rx:BehaviorSubject">\n'
            '        <Name>NetworkInput</Name>\n'
            '      </Expression>'
        )
        edge(net_src, net_input_i)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. GRAPH CONSTRUCTION
    #    Sensors: InputLayer → [Dynamics] → [activation] (inside model)
    #    Constants: InputLayer (no dynamics)
    #    Then Linear(W) per connection, fan-in with Zip+Join+Add.
    # ══════════════════════════════════════════════════════════════════════════

    def _linear(src_end, W, name):
        """Emit CreateTensor → PropertyMapping → Linear. Returns Linear index."""
        W   = np.atleast_2d(np.asarray(W, dtype=float))
        ct  = node(
            f'      <Expression xsi:type="Combinator">\n'
            f'        <Combinator xsi:type="p1:CreateTensor">\n'
            f'          <p1:Value>{fmt_W(W)}</p1:Value>\n'
            f'          <p1:DataType>Float32</p1:DataType>\n'
            f'        </Combinator>\n'
            f'      </Expression>'
        )
        pm  = node(
            f'      <Expression xsi:type="PropertyMapping">\n'
            f'        <PropertyMappings>\n'
            f'          <Property Name="Weight" />\n'
            f'        </PropertyMappings>\n'
            f'      </Expression>'
        )
        edge(ct, pm)
        lin = node(
            f'      <Expression xsi:type="Combinator">\n'
            f'        <Combinator xsi:type="p1:Linear">\n'
            f'          <p1:Name>{name}</p1:Name>\n'
            f'          <p1:InFeatures>{W.shape[1]}</p1:InFeatures>\n'
            f'          <p1:OutFeatures>{W.shape[0]}</p1:OutFeatures>\n'
            f'          <p1:Bias>false</p1:Bias>\n'
            f'        </Combinator>\n'
            f'      </Expression>'
        )
        edge(src_end, lin, 'Source1')
        edge(pm,      lin, 'Source2')
        return lin

    # Sensors: InputLayer → [Dynamics] → [activation]
    for i, s in enumerate(active_sensors):
        il_i = node(
            f'      <Expression xsi:type="Combinator">\n'
            f'        <Combinator xsi:type="p1:InputLayer">\n'
            f'          <p1:Name>{cap(s.name)}</p1:Name>\n'
            f'          <p1:Index>{i}</p1:Index>\n'
            f'        </Combinator>\n'
            f'      </Expression>'
        )
        prev = il_i

        tr  = getattr(s, 'tau_rise',  None)
        td  = getattr(s, 'tau_decay', None)
        act = (getattr(s, 'activation', 'linear') or 'linear').lower()
        if tr is not None or td is not None:
            tr = tr if tr is not None else td
            td = td if td is not None else tr
            act_str = _ACT_MAP.get(act, 'Linear')
            ll_i = node(
                f'      <Expression xsi:type="Combinator">\n'
                f'        <Combinator xsi:type="p1:LeakyLayer">\n'
                f'          <p1:Name>{cap(s.name)}</p1:Name>\n'
                f'          <p1:TauRise>{tr}</p1:TauRise>\n'
                f'          <p1:TauDecay>{td}</p1:TauDecay>\n'
                f'          <p1:Activation>{act_str}</p1:Activation>\n'
                f'        </Combinator>\n'
                f'      </Expression>'
            )
            edge(prev, ll_i)
            prev = ll_i

        chain_end[s.name] = prev
        main_end = prev

    # Constants: InputLayer only (no dynamics inside model)
    for j, lyr in enumerate(active_consts):
        inp_idx = len(active_sensors) + j
        il_i = node(
            f'      <Expression xsi:type="Combinator">\n'
            f'        <Combinator xsi:type="p1:InputLayer">\n'
            f'          <p1:Name>{cap(lyr.name)}</p1:Name>\n'
            f'          <p1:Index>{inp_idx}</p1:Index>\n'
            f'        </Combinator>\n'
            f'      </Expression>'
        )
        chain_end[lyr.name] = il_i
        main_end = il_i

    # Layers: Linear per connection, fan-in, then layer dynamics + activation
    for lyr in active_layers:
        srcs = by_tgt.get(lyr.name, [])
        if not srcs:
            warnings.append(f"'{lyr.name}' has no connections — skipped.")
            continue
        if isinstance(lyr, (MatsuokaLayer, AdaptiveLayer)):
            warnings.append(
                f"'{lyr.name}' ({type(lyr).__name__}) has no LBP.Torch equivalent"
                f" — skipped (needs manual implementation)."
            )
            continue

        branch_ends = []
        for src_name, W in srcs:
            src_end = chain_end.get(src_name)
            if src_end is None:
                warnings.append(f"Source '{src_name}' for '{lyr.name}' not found — skipped.")
                continue
            lin_name = f'W_{cap(src_name)}_to_{cap(lyr.name)}' if len(srcs) > 1 else cap(lyr.name)
            branch_ends.append(_linear(src_end, W, lin_name))

        if not branch_ends:
            continue

        if len(branch_ends) == 1:
            combined = branch_ends[0]
        else:
            zip_i = node(
                '      <Expression xsi:type="Combinator">\n'
                '        <Combinator xsi:type="rx:Zip" />\n'
                '      </Expression>'
            )
            for k, be in enumerate(branch_ends):
                edge(be, zip_i, f'Source{k+1}')
            ja_i = node(
                f'      <Expression xsi:type="Combinator">\n'
                f'        <Combinator xsi:type="p1:JoinAdditive">\n'
                f'          <p1:Name>{cap(lyr.name)}_sum</p1:Name>\n'
                f'        </Combinator>\n'
                f'      </Expression>'
            )
            edge(zip_i, ja_i)
            combined = ja_i

        if isinstance(lyr, LeakyLayer):
            act_raw = (getattr(lyr, 'activation', 'relu') or 'relu').lower()
            act_str = _ACT_MAP.get(act_raw, 'ReLU')
            extra   = ''
            if act_str == 'LeakyReLU':
                extra = f'          <p1:NegativeSlope>0.01</p1:NegativeSlope>\n'
            elif act_str == 'ELU':
                extra = f'          <p1:ELUAlpha>1.0</p1:ELUAlpha>\n'
            ll_i = node(
                f'      <Expression xsi:type="Combinator">\n'
                f'        <Combinator xsi:type="p1:LeakyLayer">\n'
                f'          <p1:Name>{cap(lyr.name)}</p1:Name>\n'
                f'          <p1:TauRise>{lyr.tau_rise}</p1:TauRise>\n'
                f'          <p1:TauDecay>{lyr.tau_decay}</p1:TauDecay>\n'
                f'          <p1:Activation>{act_str}</p1:Activation>\n'
                f'{extra}'
                f'        </Combinator>\n'
                f'      </Expression>'
            )
            edge(combined, ll_i)
            combined = ll_i

        else:
            # SumLayer (and any future layer types): apply activation if not linear
            act = (getattr(lyr, 'activation', 'linear') or 'linear').lower()
            bonsai_act = _ACT_MAP.get(act)
            if bonsai_act and bonsai_act != 'Linear':
                act_i = node(
                    f'      <Expression xsi:type="Combinator">\n'
                    f'        <Combinator xsi:type="p1:{bonsai_act}">\n'
                    f'          <p1:Name>{cap(lyr.name)}_act</p1:Name>\n'
                    f'        </Combinator>\n'
                    f'      </Expression>'
                )
                edge(combined, act_i)
                combined = act_i

        chain_end[lyr.name] = combined
        main_end = combined

    # CreateTorchModel → ModelCreated
    ctm_i = node(
        f'      <Expression xsi:type="Combinator">\n'
        f'        <Combinator xsi:type="p1:CreateTorchModel">\n'
        f'          <p1:ModelName>{model_name}</p1:ModelName>\n'
        f'        </Combinator>\n'
        f'      </Expression>'
    )
    if main_end is not None:
        edge(main_end, ctm_i)
    mc_i = node(
        '      <Expression xsi:type="rx:BehaviorSubject">\n'
        '        <Name>ModelCreated</Name>\n'
        '      </Expression>'
    )
    edge(ctm_i, mc_i)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. FORWARD PASS
    #    NetworkInput → TorchModelForward → Output → SubscribeWhen(ModelCreated)
    # ══════════════════════════════════════════════════════════════════════════

    if cl_out is not None:
        ni_sub_i = node(
            '      <Expression xsi:type="SubscribeSubject">\n'
            '        <Name>NetworkInput</Name>\n'
            '      </Expression>'
        )
        fwd_i = node(
            f'      <Expression xsi:type="Combinator">\n'
            f'        <Combinator xsi:type="p1:TorchModelForward">\n'
            f'          <p1:ModelName>{model_name}</p1:ModelName>\n'
            f'          <p1:Dt>0.02</p1:Dt>\n'
            f'        </Combinator>\n'
            f'      </Expression>'
        )
        edge(ni_sub_i, fwd_i)
        out_i = node(
            f'      <Expression xsi:type="rx:BehaviorSubject">\n'
            f'        <Name>{model_name}Output</Name>\n'
            f'      </Expression>'
        )
        edge(fwd_i, out_i)
        mc_sub_i = node(
            '      <Expression xsi:type="SubscribeSubject">\n'
            '        <Name>ModelCreated</Name>\n'
            '      </Expression>'
        )
        sw_i = node(
            '      <Expression xsi:type="Combinator">\n'
            '        <Combinator xsi:type="rx:SubscribeWhen" />\n'
            '      </Expression>'
        )
        edge(out_i,    sw_i, 'Source1')
        edge(mc_sub_i, sw_i, 'Source2')

    # ── Warning annotations ───────────────────────────────────────────────────

    for w in warnings:
        node(
            f'      <Expression xsi:type="Annotation">\n'
            f'        <Name>ExportWarning</Name>\n'
            f'        <Text><![CDATA[{w}]]></Text>\n'
            f'      </Expression>'
        )

    # ── Assemble XML ──────────────────────────────────────────────────────────

    lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        '<WorkflowBuilder Version="2.9.0"',
        '                 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"',
        '                 xmlns:rx="clr-namespace:Bonsai.Reactive;assembly=Bonsai.Core"',
        '                 xmlns:p1="clr-namespace:LBP.Torch;assembly=LBP.Torch"',
        '                 xmlns="https://bonsai-rx.org/2018/workflow">',
        '  <Workflow>',
        '    <Nodes>',
    ] + nodes_xml + [
        '    </Nodes>',
        '    <Edges>',
    ] + edges_xml + [
        '    </Edges>',
        '  </Workflow>',
        '</WorkflowBuilder>',
    ]
    return '\n'.join(lines)
