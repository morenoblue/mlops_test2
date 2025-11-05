# myproj — real-data smoke sample (no LFS)

## Workflow
1) **Locally**, run `scripts/make_smoke_sample.py` to create a *tiny* Parquet from your real dataset:
   ```bash
   python scripts/make_smoke_sample.py --source /abs/path/to/your/data_or_dir \
     --rows 256 --out data/smoke_sample.parquet --seed 0

