"""Fetch the official checkpoint archive and extract only its Paper ViT-L/14 assets."""
import argparse
from pathlib import Path
import zipfile

from tacticalai.export import sha256, write_json
from tacticalai.clip_reident import UPSTREAM_COMMIT

ARCHIVE_ID = '1Gm5J19okhLdnZTQLUsjfYoI0rwrLQ09i'
PREFIX = 'model/ViT-L-14_openai/Paper/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('models/clip-reident'))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    archive = args.root / 'checkpoints.zip'
    if not archive.exists():
        import gdown
        gdown.download(id=ARCHIVE_ID, output=str(archive), quiet=False, resume=True)
    with zipfile.ZipFile(archive) as bundle:
        for name in ('weights_e4.pth', 'config.py', 'result_server.txt', 'log.txt'):
            member = PREFIX + name
            # Extract explicit fixed members, never arbitrary archive paths.
            destination = args.root / member
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                with bundle.open(member) as src, destination.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src, dst)
    checkpoint = args.root / PREFIX / 'weights_e4.pth'
    write_json(args.root / 'provenance.json', dict(
        upstream='https://github.com/DeepSportradar/2022-winners-player-reidentification-challenge',
        upstream_commit=UPSTREAM_COMMIT,
        archive_url=f'https://drive.google.com/file/d/{ARCHIVE_ID}/view',
        archive_sha256=sha256(archive), checkpoint_member=PREFIX + 'weights_e4.pth',
        checkpoint_sha256=sha256(checkpoint),
        license='Upstream code MIT; checkpoint/dataset commercial terms not independently established.'))
    print(checkpoint.resolve())


if __name__ == '__main__':
    main()
