# Deployment

The backend runs on **Render** (free Docker instance, 512 MB), the frontend on **Vercel**.
Both deploy automatically from the `main` branch of the GitHub repository.

## 1. Backend on Render

Service settings (Settings tab):

| Setting | Value |
|---|---|
| Runtime | Docker |
| Root Directory | `backend` |
| Dockerfile Path | `./Dockerfile` |
| Health Check Path | `/health` |
| Branch | `main` |

Environment variables (Environment tab):

| Variable | Value |
|---|---|
| `ALLOWED_ORIGINS` | The Vercel production URL, e.g. `https://my-app.vercel.app` (no trailing slash; several: comma-separated) |
| `ALLOWED_ORIGIN_REGEX` | Optional, to allow Vercel preview deployments, e.g. `https://my-app-.*\.vercel\.app` |

Delete `API_KEY` if it still exists: the API no longer uses a key. Other limits
(`MAX_ROWS`, `TRAIN_RATE_LIMIT_PER_MINUTE`...) have sensible defaults, see README.

Memory: measured peak ~380 MB when training and forecasting with the five models
on 3,650 rows, for a 512 MB limit.

## 2. Frontend on Vercel

Project settings: Framework Next.js, Root Directory `frontend`.

Environment variables (Settings > Environment Variables, Production and Preview):

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | The Render URL, e.g. `https://time-series-forecaster-api.onrender.com` |

Delete the variables of the former API key setup if they still exist:
`NEXT_PUBLIC_API_KEY`, `NEXT_PUBLIC_API_MODE`, `BACKEND_PRIVATE_URL`, `BACKEND_API_KEY`.
`NEXT_PUBLIC_*` variables are embedded at build time: redeploy after changing them.

## 3. Deploy

Merge into `main` and push: Render rebuilds the Docker image (a few minutes the first
time), Vercel rebuilds the frontend.

## 4. Check

1. `https://<render-url>/health` answers `{"status":"healthy"}`.
2. Open the Vercel URL. After 15 minutes without visits the backend sleeps: a
   "Waking up the server…" banner shows for about a minute, then disappears.
3. Load the "Daily minimum temperatures" example, add the recommended models, train,
   save a model to the library, forecast in the Forecast space.
4. Browser DevTools > Console: no CORS error.

## Troubleshooting

| Symptom | Cause |
|---|---|
| CORS error in the console | `ALLOWED_ORIGINS` does not contain the exact origin of the page (scheme, domain, no trailing slash) |
| "The server cannot be reached" | Wrong `NEXT_PUBLIC_API_URL`, or the Render service is down (check its logs) |
| Render log "Ran out of memory" | Dataset or models too large for 512 MB: lower `MAX_ROWS` or `MAX_MODELS` |
| 429 errors | Rate limits per client IP (`TRAIN_RATE_LIMIT_PER_MINUTE`, `ANALYZE_RATE_LIMIT_PER_MINUTE`) |
