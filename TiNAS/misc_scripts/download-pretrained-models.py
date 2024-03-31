import base64
import hashlib
import os.path
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
print(sys.path)

import wandb

from settings import Settings

WANDB_PRETRAINED_MODEL_DIR = '<ADD_DIR>'

def fix_run_tags(runs):
    # add new tags to existing runs
    for run in runs:
        # https://docs.wandb.ai/guides/app/features/tags
        cur_tags = list(run.tags)
        print(cur_tags)
        if 'SUPERNET_TRAINING' not in run.tags:
            run.tags.append('SUPERNET_TRAINING')
        if set(cur_tags) != set(run.tags):
            print(cur_tags)
            print(run.tags)
            # https://docs.wandb.ai/ref/python/public-api/run
            # run.update()

def main():
    global_settings = Settings()
    checkpoint_dir = global_settings.NAS_SETTINGS_GENERAL['CHECKPOINT_DIR']

    api = wandb.Api()
    runs = api.runs(
        path= WANDB_PRETRAINED_MODEL_DIR,
        # filter by tags https://github.com/wandb/wandb/issues/2699
        # filters={"tags": {"$in": ["TiNAS-U.json"]}}
    )

    # List and download files https://github.com/wandb/wandb/issues/5641
    for run in runs:
        if run.state != 'finished':
            print(f'Skip ongoing or failed run {run.tags!r}')
            continue

        for file in run.files():
            # Just downloading pre-trained models
            if not file.name.endswith('.pth'):
                continue

            downloaded_path = os.path.join(checkpoint_dir, file.name)

            if os.path.exists(downloaded_path):
                with open(downloaded_path, 'rb') as f:
                    local_file_md5 = base64.b64encode(hashlib.md5(f.read()).digest()).decode('ascii')

                if file.md5 != local_file_md5:
                    print(f'ERROR: local file {downloaded_path} exists and is different from the remote file')
                else:
                    print(f'{downloaded_path} is already downloaded, skipping')

                continue

            file.download(root=checkpoint_dir)

if __name__ == '__main__':
    main()
