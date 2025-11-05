# image-first CI with real-data smoke (flat src layout)

- Everything is under `src/`
- The pipeline **defaults** to `src/data/smoke_sample.parquet` (module-relative)
- Generate that file with `src/scripts/make_smoke_sample.py` (edit constants at top, then run)
- CI builds the image, smoke-runs it, and pushes the same image to GHCR on non-PR runs

## Create the tiny smoke sample (once, or whenever you need to refresh)
Edit `SRC_SOURCE_PATH` in `src/scripts/make_smoke_sample.py` to point at your dataset (file or dir), then:
```bash
python src/scripts/make_smoke_sample.py
git add src/data/smoke_sample.parquet
git commit -m "Add/refresh smoke sample"
git push

