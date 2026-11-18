"""Render a saved TRACE term offline. Never invokes the standardization pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
from decision_trace_view import TraceBundle, build_model
from decision_trace_renderer import render


def visualize(directory,term_id,output,*,include_details=False,export_format='png'):
    if export_format not in ('png','svg','both'):raise ValueError('Unsupported export format')
    source=Path(directory).resolve();output=Path(output).resolve()
    # A separate destination protects the input bundle from renderer side effects.
    if output==source or source in output.parents or output in source.parents:
        raise ValueError('Output must be separate from the source run directory')
    if output.exists():raise FileExistsError('Choose a fresh visualization output directory')
    bundle=TraceBundle(source,term_id)
    model=build_model(bundle)
    output.mkdir(parents=True,exist_ok=False)
    files=render(model,output,include_details,export_format)
    event_path=source/'traces'/'events'/(term_id+'.jsonl')
    manifest={'renderer_version':'trace.static-view/2','source_directory':str(source),
        'render_options':{'include_details':include_details,'export_format':export_format},
        'source_events_sha256':hashlib.sha256(event_path.read_bytes()).hexdigest() if event_path.exists() else None,
        'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__),Path(__file__).with_name('decision_trace_view.py'),Path(__file__).with_name('decision_trace_renderer.py'),
            Path(__file__).with_name('trace_rasterize.ps1'),Path(__file__).with_name('validate_decision_trace.py'))},
        **model,'generated_files':[{'path':f,'sha256':hashlib.sha256((output/f).read_bytes()).hexdigest()} for f in files]}
    with (output/'rendering_manifest.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2,sort_keys=True)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--term-id',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--include-details',action='store_true',help='Also render numbered evidence-detail pages')
    p.add_argument('--export-format',choices=('png','svg','both'),default='png',help='Image format (default: png)')
    a=p.parse_args()
    result=visualize(a.directory,a.term_id,a.output,include_details=a.include_details,export_format=a.export_format)
    print(json.dumps({'output':str(a.output),'term_id':a.term_id,'trace':result['validation']['trace_completeness'],
        'mapping':result['mapping_status'],'files':len(result['generated_files']),
        'warning_count':len(result['findings'])}))
    # Incomplete history can still render successfully. Findings are never hidden.
    return 0


if __name__=='__main__':raise SystemExit(main())
