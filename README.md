Test:
```sh
uv run python src/data_processor.py test_config.yaml --output-dir processed_data;
uv run python src/data_reconstructor.py --output-dir reconstructed
```

Math:
```sh
uv run python src/data_processor.py math_config.yaml --output-dir processed_data;
```

All: 
```sh
uv run python src/data_processor.py config.yaml --output-dir processed_data --upload-hf --delete-after-upload;
```

Reconstruct:
```sh
uv run python src/data_reconstructor.py test_processed_data --output-dir reconstructed
```

Upload to HF:
```sh
huggingface-cli login
huggingface-cli upload aspisov/olmo-data . . --repo-type dataset
```