"""Verify the downloaded dataset and included model/result files."""

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-only', action='store_true', help='Check included artifacts without the sensor dataset.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = {}
    pretrained = json.loads((root / 'pretrained_models.json').read_text())
    for model in pretrained['models']:
        for name, record in model['files'].items():
            expected[f"{model['path']}/{name}"] = record['sha256']
    for folder, filename in [('results/single_vg', 'archive_manifest.json'),
                              ('results/analysis_reference/chemical_humid_shap', 'manifest.json')]:
        manifest = json.loads((root / folder / filename).read_text())
        for name, record in manifest['files'].items():
            expected[f'{folder}/{name}'] = record if isinstance(record, str) else record['sha256']
    errors = []
    if not args.code_only:
        dataset = json.loads((root / 'dataset_manifest.json').read_text())
        expected.update({name: record['sha256'] for name, record in dataset['files'].items()})
        actual = {p.relative_to(root).as_posix() for p in (root / 'TBDT_Dataset').rglob('*.npy')}
        for name in sorted(actual - set(dataset['files'])):
            errors.append(f'Unexpected dataset file: {name}')
    for name, expected_digest in expected.items():
        path = root / name
        if not path.is_file():
            errors.append(f'Missing file: {name}')
        elif digest(path) != expected_digest:
            errors.append(f'Checksum mismatch: {name}')
    if errors:
        for message in errors[:10]:
            print(message)
        if len(errors) > 10:
            print(f'...and {len(errors) - 10} more errors.')
        print('Download and extract the code and dataset as described in README.md.')
        raise SystemExit(1)
    print(f'Verified {len(expected)} files: all SHA-256 checksums match.')


if __name__ == '__main__':
    main()
