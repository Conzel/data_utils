
#!/usr/bin/env bash
rsync --progress --exclude ".DS_Store" --exclude ".pytest_cache" --exclude "__pycache__/" --exclude "*egg" --exclude "wandb/" --exclude "build" --exclude "artifacts" -r src pyproject.toml requirements.txt galvani:~/data_utils
