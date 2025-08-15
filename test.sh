clear
rm -rf processed_data/
rm -rf reconstructed_data/
uv run python src/data_processor.py \
    configs/test_config.yaml \
    --upload-hf \
    --hf-repo aspisov/dataset \
    --hf-folder test \
    --max-lines 100000 \
    --max-mb 3.0 \
    # --delete-after-upload 

uv run python src/data_reconstructor.py \
    processed_data/ \
    --output reconstructed_data/ \

uv run python scripts/compare_hashes.py \
    processed_data/file_hashes.json \
    reconstructed_data/file_hashes.json