"""Read-only exact component spans from the already imported official LOINC.

Part names identify alternative component concepts, not fully specified tests.
No source axes are imputed to the reported input; no overlap is silently dropped.
"""
from runtime_resources import resource as asp_resource
import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

class ComponentSpans:

    def __init__(self, directory):
        directory = Path(directory)
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        db = directory / 'loinc.sqlite'
        with db.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if not manifest.get('complete_import') or actual != manifest['database_sha256']:
            raise ValueError('LOINC import incomplete or changed')
        self.binding = dict(release=manifest['release'], database_sha256=actual, package_sha256=manifest['package_sha256'])
        self.names = {}
        with closing(sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True)) as connection:
            rows = connection.execute("SELECT DISTINCT p.code,p.raw FROM parts p JOIN links l ON l.part=p.code JOIN terms t ON t.code=l.term WHERE p.kind='COMPONENT' AND p.status='ACTIVE' AND t.status='ACTIVE' AND l.kind='Primary'")
            for code, raw in rows:
                record = json.loads(raw)
                for field in ('PartName', 'PartDisplayName'):
                    name = record.get(field, '')
                    key = self.tokens(name)
                    if key:
                        item = dict(source='LOINC Component Part', code=code, field=field, official_name=name, **self.binding)
                        self.names.setdefault(key, []).append(item)
        self.max_words = max(map(len, self.names), default=0)

    @staticmethod
    def tokens(text):
        return tuple(re.findall('\\w+|[^\\w\\s-]', text.casefold()))

    def spans(self, text):
        words = list(re.finditer('\\w+|[^\\w\\s-]', text))
        values = [w[0].casefold() for w in words]
        matches = []
        for start in range(len(words)):
            for end in range(start + 1, min(len(words), start + self.max_words) + 1):
                witnesses = self.names.get(tuple(values[start:end]))
                if witnesses:
                    matches.append(dict(text=text[words[start].start():words[end - 1].end()], span=[words[start].start(), words[end - 1].end()], witnesses=witnesses, candidate_concepts=sorted({w['code'] for w in witnesses})))
        for match in matches:
            overlap = [m for m in matches if m is not match and max(m['span'][0], match['span'][0]) < min(m['span'][1], match['span'][1]) and (m['candidate_concepts'] != match['candidate_concepts'])]
            match['status'] = 'ambiguous_component_or_nested_scope' if overlap or len(match['candidate_concepts']) != 1 else 'exact_component_name'
        return matches
