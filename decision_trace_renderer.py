"""Deterministic static SVG/PNG pages from a compact trace presentation model.

SVG uses standard-library XML; PNG uses Windows' local System.Drawing, with no
network or external rendering service. Both consume the same drawing primitives.
"""
import json
import math
from pathlib import Path
import subprocess
import textwrap
import xml.etree.ElementTree as ET
from decision_trace_view import readable

WIDTH=1440
HEIGHT=1700
INK='#233344'
MUTED='#536579'
COLORS={'model':'#edf3fc','computed':'#edf5f2','warning':'#fff3db',
        'repair':'#fff3db','final':'#dfeee9','candidate':'#f0f1f7','observed':'#f4f6f8'}


def wrapped(text, width, size=19):
    # Conservative font-width allowance; all content retains full references.
    columns=max(12,int(width/(size*.61)))
    lines=[]
    for para in str(text).splitlines() or ['']:
        lines.extend(textwrap.wrap(para,columns,break_long_words=True,break_on_hyphens=False) or [''])
    return lines


class Canvas:
    def __init__(self,title,subtitle,model,width=WIDTH,height=HEIGHT,overview=False):
        self.items=[]
        self.title=title
        self.width=width;self.height=height
        self.rect(0,0,width,height,'#ffffff','#ffffff')
        self.rect(0,0,width,8,'#28695f','#28695f')
        if overview:return
        self.text(44,34,title,28,True)
        self.text(44,76,subtitle,18,False,MUTED)
        status=model['validation']['trace_completeness']
        self.rect(44,110,WIDTH-88,62,COLORS['warning'] if status!='complete' else COLORS['computed'],'#b8c9c5')
        self.text(58,120,'TRACE: '+status.upper()+'   |   MAPPING: '+str(model['mapping_status'])+'   |   '+
                  ('SYNTHETIC FIXTURE' if model['synthetic'] else 'SAVED EVIDENCE'),18,True)
        self.text(58,147,'Execution status, mapping outcome and trace integrity are different. No clinical correctness is inferred.',17,False,MUTED)
        self.text(44,HEIGHT-56,'Evidence keys, exact event IDs and immutable artifact references: rendering_manifest.json',17,False,MUTED)
        self.text(44,HEIGHT-30,'TRACE | Offline static view | Recorded model text is a claim; scores are not probabilities.',16,False,MUTED)

    def rect(self,x,y,w,h,fill,stroke=INK):
        self.items.append(dict(op='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke))
    def text(self,x,y,text,size=19,bold=False,color=INK):
        self.items.append(dict(op='text',x=x,y=y,text=str(text),size=size,bold=bold,color=color))
    def line(self,points,color=MUTED,arrow=False,dashed=False):
        self.items.append(dict(op='line',points=points,color=color,arrow=arrow,dashed=dashed))


def source_panel_layout(panel,width):
    """Measured text inside a source box, not an additional workflow node."""
    lines=[];y=12
    def add(text,size=18,bold=False):
        nonlocal y
        for line in wrapped(text,width-24,size):
            lines.append((12,y,line,size,bold));y+=24
    add(panel['title'],20,True)
    add('Rank applies within this source only.',16)
    y+=8
    if not panel['entries']:add(panel['empty_message'])
    queries={e.get('query') for e in panel['entries'] if e.get('query')}
    for entry in panel['entries']:
        rank='Rank '+str(entry['rank']) if entry['rank'] is not None else 'Rank unavailable'
        if panel['source']=='fuzzy':
            score=entry.get('ranking_score')
            rank+=' · Ranking score '+(format(score,'.6g') if score is not None else 'unavailable')
        elif entry['score'] is not None:
            rank+=' · Cosine '+format(entry['score'],'.6g')
        add(rank,18,True)
        if panel['source']=='fuzzy':
            score=entry['score']
            add('Raw fuzzy similarity: '+(format(score,'.6g') if score is not None else 'unavailable'),17)
        add('LLT: '+entry['llt_name']+' ('+str(entry['candidate_id'] or 'identity unavailable')+')')
        add('PT: '+entry['pt_name']+' (PT:'+str(entry['pt_code'] or '?')+')')
        if len(queries)>1:add('Query: '+str(entry.get('query') or 'unavailable'),16)
        for note in entry['notes']:add(note,16)
        y+=10
    return {'lines':lines,'height':y+4,'width':width}


def overview_pages(model,include_details=False,extension='png'):
    # Entry order is a stable layout convention. Only recorded dependencies draw
    # arrows. Parallel graph retrieval sources share one row, regardless of their
    # physical finish order in the event stream.
    nodes=model['nodes']; rows=[]; index=0
    while index<len(nodes):
        n=nodes[index]
        if n['stage'] in ('fuzzy_matching','semantic_matching','umls_matching'):
            group=[]
            while index<len(nodes) and nodes[index]['stage'] in ('fuzzy_matching','semantic_matching','umls_matching'):
                group.append(nodes[index]);index+=1
            rows.append(sorted(group,key=lambda x:('fuzzy_matching','semantic_matching','umls_matching').index(x['stage'])))
        else:
            rows.append([n]);index+=1
    # Size from rendered content. A bounded line length preserves readability;
    # there is no fixed page height, node height, or overview pagination.
    max_parallel=max([len(row) for row in rows] or [1])
    longest=max([len(text) for n in nodes for text in n.get('overview_lines',[])]+[55])
    # Expand to the needed reading width, with a line-length cap. Parallel
    # columns set a minimum, not a fixed page size.
    width=int(max(760,min(1120,148+longest*11),120+max_parallel*360+(max_parallel-1)*20))
    inner=width-120; coords={}; bodies={}; headers={}; panel_layouts={}
    subtitle=(model.get('original_input') or 'Original input unavailable')+' · '+('SYNTHETIC FIXTURE' if model['synthetic'] else 'Saved evidence')
    subtitle_lines=wrapped(subtitle,width-80,17)
    banner_y=60+len(subtitle_lines)*23+3
    warnings=[]
    if model['findings']:
        if any('dependency' in f or 'handoff' in f or 'upstream' in f for f in model['findings']):
            warnings.append('Dependency gap: upstream execution is not fully linked to completion. No missing arrows are invented.')
        if any('missing' in f or 'corrupt' in f or 'hash' in f or 'unsupported' in f for f in model['findings']):
            warnings.append('Some evidence is missing, inconsistent or unsupported; inspect the manifest for exact findings.')
        warnings.append(str(len(model['findings']))+' validation findings retained in the manifest; the mapping result is separate.')
    banner_lines=['Trace: '+model['validation']['trace_completeness'].upper()+
                  '  |  Final mapping: '+readable(model['mapping_status'])]+warnings
    banner_lines=[line for p in banner_lines for line in wrapped(p,width-100,17)]
    banner_height=20+len(banner_lines)*23
    y=banner_y+banner_height+66
    for row in rows:
        column_width=(inner-(len(row)-1)*20)/len(row)
        row_height=0
        for col,n in enumerate(row):
            title=n['id']+'  '+n['title']+(' · round '+str(n['round']) if n['round'] else '')
            paras=n.get('overview_lines') or ['Recorded stage '+readable(n['execution_status'])+'.']
            if include_details and n['detail_ids']:
                paras=paras+['Evidence cards: '+', '.join(n['detail_ids'])]
            preferred=max([len(t)*11+30 for t in paras]+[len(title)*12.2+30])
            node_width=column_width if len(row)>1 else min(inner,max(460,preferred))
            panels=n.get('candidate_source_panels',[])
            if panels:node_width=column_width
            body=[line for p in paras for line in wrapped(p,node_width-28,18)]
            heading=wrapped(title,node_width-28,20)
            h=22+len(heading)*26+22+len(body)*24
            if panels:
                panel_width=(node_width-28-(len(panels)-1)*16)/len(panels)
                specs=[]
                for pi,panel in enumerate(panels):
                    spec=source_panel_layout(panel,panel_width)
                    specs.append({**spec,'x':14+pi*(panel_width+16),'y':h+6,'panel':panel})
                panel_layouts[n['id']]=specs
                h+=20+max(p['height'] for p in specs)
            x=60+col*(column_width+20) if len(row)>1 else (width-node_width)/2
            coords[n['id']]=(x,y,node_width,h)
            bodies[n['id']]=body;headers[n['id']]=heading
            row_height=max(row_height,h)
        y+=row_height+30
    height=int(y+48)
    c=Canvas('TRACE decision flow',subtitle,model,width,height,overview=True)
    c.text(40,24,'TRACE decision flow',28,True)
    for i,line in enumerate(subtitle_lines):c.text(40,60+i*23,line,17,False,MUTED)
    status=model['validation']['trace_completeness']
    c.rect(40,banner_y,width-80,banner_height,COLORS['warning'] if status!='complete' else COLORS['computed'],'#b8c9c5')
    for i,line in enumerate(banner_lines):c.text(52,banner_y+10+i*23,line,17,i==0)
    c.text(40,banner_y+12+banner_height,'Legend: Blue = interpretation/review  |  Green = pipeline stages  |  Amber = warnings',16,False,MUTED)
    c.text(40,banner_y+35+banner_height,'Arrows show recorded dependencies only. Completed execution does not mean accepted or clinically correct.',16,False,MUTED)
    # Recorded edges first so they never obscure text inside the node.
    for edge in model['edges']:
        a=coords[edge['from']];b=coords[edge['to']]
        # Adjacent rows route through their shared gap; distant dependencies use
        # outer lanes. Neither layout nor historical adjacency creates an edge.
        sx=a[0]+a[2]/2;tx=b[0]+b[2]/2;sy=a[1]+a[3];ty=b[1]
        between=any(sy<pos[1]<ty for pos in coords.values())
        if ty>sy and not between:
            bend=ty-15
            c.line([(sx,sy),(sx,bend),(tx,bend),(tx,ty)],arrow=True)
        else:
            lane=18+(int(edge['from'][1:])%4)*7
            c.line([(a[0],a[1]+a[3]/2),(lane,a[1]+a[3]/2),(lane,b[1]+b[3]/2),(b[0],b[1]+b[3]/2)],arrow=True)
    model['overview_panel_locations']={}
    for n in nodes:
        x,y,w,h=coords[n['id']]
        kind='final' if n['stage']=='final' else 'model' if n['stage'] in ('input_interpretation','reviewer_llm') else 'warning' if n['stage']=='completion.handoff' else 'computed'
        c.rect(x,y,w,h,COLORS[kind],'#81968f' if kind!='warning' else '#ad8031')
        line_y=y+10
        for line in headers[n['id']]:c.text(x+14,line_y,line,20,True);line_y+=26
        attribution=' · MODEL / returned evidence' if kind=='model' else ' · SAVED RESULT' if kind=='final' else ''
        c.text(x+14,line_y,'Execution '+readable(n['execution_status'])+attribution,15,False,MUTED);line_y+=24
        for line in bodies[n['id']]:
            c.text(x+14,line_y,line,18);line_y+=24
        for spec in panel_layouts.get(n['id'],[]):
            px=x+spec['x'];py=y+spec['y'];panel=spec['panel']
            c.rect(px,py,spec['width'],spec['height'],'#ffffff','#a7b9b5')
            for tx,ty,text,size,bold in spec['lines']:c.text(px+tx,py+ty,text,size,bold)
            model['overview_panel_locations'][n['id']+':'+panel['source']]={
                'file':'overview.'+extension,'node_id':n['id'],'source':panel['source'],
                'bounds':[px,py,spec['width'],spec['height']],
                'candidate_ids':[entry['candidate_id'] for entry in panel['entries']],
                **n.get('overview_sources',{})}
    model['overview_locations']={k:{'file':'overview.'+extension,'node_id':k,'bounds':list(v)} for k,v in coords.items()}
    model['overview_canvas']={'width':width,'height':height}
    return [('overview',c)]


def detail_pages(model,extension='png'):
    pages=[]; current=None;y=0; locations={}
    def new_page():
        nonlocal current,y
        num=len(pages)+1
        current=Canvas(f'TRACE evidence details {num}',model['term_id']+' | Refer to numbered overview nodes and detail IDs',model)
        pages.append((f'details_{num:02d}',current));y=202
    for card in model['cards']:
        lines=[]
        for para in card['lines']:
            lines+=wrapped(para,WIDTH-140,19)+['']
        refs='Events: '+', '.join(e.split(':')[-1] for e in card['event_ids'])+' | Evidence: '+', '.join(card['artifact_ids'])
        ref_lines=wrapped(refs,WIDTH-140,16)
        if card.get('json_pointer'):ref_lines+=wrapped('JSON pointer: '+card['json_pointer'],WIDTH-140,16)
        offset=0;part=0
        while offset<len(lines) or part==0:
            if current is None or y>HEIGHT-340:new_page()
            capacity=int((HEIGHT-110-y-100)/24)
            if capacity<4:new_page();capacity=int((HEIGHT-110-y-100)/24)
            take=min(capacity,len(lines)-offset)
            chunk=lines[offset:offset+take]
            # Long reference inventories remain complete in manifest, compact on
            # paper. This is display condensation, not removal of evidence.
            visible_refs=ref_lines[:3]
            if len(ref_lines)>3:visible_refs[-1]='Additional references: rendering_manifest.json, element '+card['id']
            height=58+len(chunk)*24+len(visible_refs)*20+12
            if y+height>HEIGHT-100:
                take=max(1,take-4);chunk=lines[offset:offset+take]
                height=58+len(chunk)*24+len(visible_refs)*20+12
            current.rect(44,y,WIDTH-88,height,COLORS.get(card['kind'],COLORS['computed']),'#c0c9d0')
            head=card['id']+' | '+card['title']+(' (continued)' if part else '')
            current.text(60,y+12,head[:108],21,True)
            line_y=y+48
            for text in chunk:
                current.text(60,line_y,text,19);line_y+=24
            for text in visible_refs:
                current.text(60,line_y,text,16,False,MUTED);line_y+=20
            locations.setdefault(card['id'],[]).append(pages[-1][0]+'.'+extension)
            y+=height+22;offset+=take;part+=1
    model['detail_locations']=locations
    return pages


def svg_bytes(canvas):
    ns='http://www.w3.org/2000/svg'
    ET.register_namespace('',ns)
    root=ET.Element('{'+ns+'}svg',{'width':str(canvas.width),'height':str(canvas.height),'viewBox':f'0 0 {canvas.width} {canvas.height}',
        'role':'img','aria-label':canvas.title})
    ET.SubElement(root,'title').text=canvas.title
    for item in canvas.items:
        if item['op']=='rect':
            ET.SubElement(root,'rect',{k:str(item[k]) for k in ('x','y','fill','stroke')}|{'width':str(item['w']),'height':str(item['h']),'stroke-width':'1'})
        elif item['op']=='text':
            el=ET.SubElement(root,'text',{'x':str(item['x']),'y':str(item['y']+item['size']),
                'font-family':'Segoe UI, Arial, sans-serif','font-size':str(item['size']),
                'font-weight':'600' if item['bold'] else '400','fill':item['color']})
            el.text=item['text']
        else:
            attrs={'points':' '.join(f'{x},{y}' for x,y in item['points']),'stroke':item['color'],'stroke-width':'2','fill':'none'}
            if item['dashed']:attrs['stroke-dasharray']='6 4'
            ET.SubElement(root,'polyline',attrs)
            if item['arrow']:
                a,b=item['points'][-2:];angle=math.atan2(b[1]-a[1],b[0]-a[0])
                points=[b,(b[0]-9*math.cos(angle-.45),b[1]-9*math.sin(angle-.45)),(b[0]-9*math.cos(angle+.45),b[1]-9*math.sin(angle+.45))]
                ET.SubElement(root,'polygon',{'points':' '.join(f'{x:.2f},{y:.2f}' for x,y in points),'fill':item['color']})
    return ET.tostring(root,encoding='utf-8',xml_declaration=True)


def render(model, output,include_details=False,export_format='png'):
    if export_format not in ('png','svg','both'):raise ValueError('Unsupported export format')
    extension='svg' if export_format=='svg' else 'png'
    model['detail_locations']={}
    pages=overview_pages(model,include_details,extension)
    if include_details:pages+=detail_pages(model,extension)
    suffixes=('.png','.svg') if export_format=='both' else ('.'+export_format,)
    scenes=[]
    for stem,canvas in pages:
        if '.svg' in suffixes:
            with (output/(stem+'.svg')).open('xb') as f:f.write(svg_bytes(canvas))
        scenes.append({'filename':stem+'.png','width':canvas.width,'height':canvas.height,'items':canvas.items})
    # Only a compact drawing specification is temporary; never copy snapshots.
    if '.png' in suffixes:
        scene=output/'render_scene.json'
        with scene.open('x',encoding='utf-8') as f:json.dump(scenes,f,ensure_ascii=False,separators=(',',':'))
        helper=Path(__file__).with_name('trace_rasterize.ps1')
        subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(helper),
                        '-Scene',str(scene),'-Directory',str(output)],check=True,capture_output=True,text=True)
        scene.unlink()  # Only our newly created temporary rendering specification.
    return [stem+suffix for stem,_ in pages for suffix in suffixes]
