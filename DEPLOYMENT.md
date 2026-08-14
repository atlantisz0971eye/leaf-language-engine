# Leaf Language Engine deployment

The production app is a single Docker service. Source updates and runtime data
are intentionally separated:

- Git stores the application, language database, UI, and compact curated seeds.
- A Railway volume mounted at `/data` stores uploads, analyses, and exports.
- Every push to the connected `main` branch triggers a new deployment.
- Existing files in `/data` survive deployments and restarts.

## First Railway deployment

1. Run `python3 scripts/prepare_deploy_seed.py` and commit the resulting
   `seed_data` folder.
2. Push this folder to the connected GitHub repository.
3. In Railway, create a service from that repository. Railway will detect the
   root `Dockerfile` and deploy it.
4. Attach one volume to the service with mount path `/data`.
5. Add the service variable `LEAF_DATA_DIR=/data`.
6. Keep `LEAF_MAX_CONCURRENT_JOBS=1` while the service has limited memory.
7. On the Railway Hobby service, set `LEAF_MAX_RUBBING_RESOLUTION=4096`.
8. Keep `LEAF_ENABLE_NMNS_PIPELINE=0` for the public pixel-only release.
9. Generate a Railway domain and open `/health`. A healthy response reports
   `"status": "ok"`.

The first start copies the curated NMNS and rubbing examples into an empty
volume. Later deployments add missing seeds but never overwrite user data.

## Normal updates

1. Test locally with `python3 app.py`.
2. Refresh bundled curated examples only when required:
   `python3 scripts/prepare_deploy_seed.py`.
3. Commit and push to `main`.
4. Railway builds and publishes the new version automatically. The `/data`
   volume remains unchanged.

## Small-team operating limits

- Use one Railway service instance because background job state is currently
  held in process memory.
- The website and APIs are public. Do not upload private or sensitive material
  in specimens, text, filenames, or generated archives.
- The public release exposes only the leaf-rubbing pixel pipeline. Keep the
  maximum export resolution proportional to the memory available to the service.
- WAV feature extraction runs in the browser. Only the compact feature array is
  sent to the service; disturbed PNGs and reports are stored under
  `/data/analysis_output/audio_disturbance`.
- Upgrade memory or optimize the colour renderer before enabling 4096–8192 px
  leaf-rubbing exports online.
- Back up the Railway volume before changing codecs or the stored data layout.
