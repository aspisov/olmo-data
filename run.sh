
uv run python src/data_processor.py \
    configs/dima_config.yaml \
    --upload-hf \
    --hf-repo aspisov/dataset \
    --hf-folder dima \
    --max-lines 500000000 \
    --max-mb 10000.0 \
    --delete-after-upload 