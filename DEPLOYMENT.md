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
2. Push this folder to a private GitHub repository.
3. In Railway, create a service from that repository. Railway will detect the
   root `Dockerfile` and deploy it.
4. Attach one volume to the service with mount path `/data`.
5. Add the service variable `LEAF_DATA_DIR=/data`.
6. Add `LEAF_ACCESS_PASSWORD` with a long shared team password. The health
   check stays public, while all pages and APIs require that password.
7. Keep `LEAF_MAX_CONCURRENT_JOBS=1` for the initial four-person test.
8. Generate a Railway domain and open `/health`. A healthy response reports
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
- Share the team password only with the four project members.
- Start with a 5 GB volume and 2 GB of memory.
- Prefer 4096 or 8192 px exports online. Treat 16384 px as an experimental,
  high-memory export.
- Back up the Railway volume before changing codecs or the stored data layout.
