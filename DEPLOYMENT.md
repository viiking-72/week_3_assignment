# Deployment Notes

How this project runs without OpenRouter credits, and what it takes to fit on
free hosting. Every number below was measured on this codebase, not estimated.

## Providers

| Concern | Service | Cost | Why |
|---|---|---|---|
| Vector search | Qdrant Cloud | free tier | 11,008 vectors, well inside limits |
| Query embedding (local) | fastembed, on your machine | free | No key, 42 ms/query |
| Query embedding (deployed) | Hugging Face Inference | free tier, no card | Keeps the model off a 512 MB host |
| Answer generation | Cohere | free trial key, no card | 1,000 calls/month caps spend |
| Reranking | Cohere | same key | |

Switch any of them with one environment variable. No code change.

### The embedding model cannot be substituted

The vectors already in Qdrant were produced by `BAAI/bge-large-en-v1.5`. Every
embedding provider above serves that same model. Swapping in a different model
gives a different vector space, and search then returns confident nonsense.
Matching the dimension count is not sufficient. Several other 1024-dimension
models exist and none of them are compatible.

## Fitting Render's free tier

Render's free plan gives 512 MB of RAM. The backend did not fit.

| Configuration | Startup peak | Fits in 512 MB |
|---|---|---|
| Original, corpus in memory | 631 MB | no |
| Plus a local embedding model | 2,433 MB | no |
| `LOW_MEMORY=true` with prebuilt index | 157 MB | yes |

Two things caused the original footprint. The chunk corpus was held in memory,
and `BM25Okapi` stores one Python dictionary of term counts per document, which
across 11,008 chunks costs hundreds of megabytes that the interpreter never
returns to the operating system.

`LOW_MEMORY=true` changes both:

- Chunk payloads are read from Qdrant, which already stores them, instead of
  being kept in memory a second time.
- Keyword search uses `CompactBM25Index`, which holds the same postings in flat
  numpy arrays. It is verified to produce the identical top-10 ordering as
  `BM25Okapi` across test queries, and the saved artifact is 3.3 MB.

Build the artifact once before deploying:

```bash
cd backend
python build_bm25_index.py --verify
```

That writes `bm25_index.npz`, which is committed deliberately. The server loads
it and never opens `chunks.json`.

## Deploying

**Backend on Render.** `backend/render.yaml` holds the configuration. Set the
four secrets in the Render dashboard rather than in the file:
`QDRANT_URL`, `QDRANT_API_KEY`, `COHERE_API_KEY`, `HUGGINGFACE_API_KEY`.

Free instances sleep after 15 minutes idle and take 30 to 60 seconds to wake,
so the first query after a pause will be slow.

**Frontend on Vercel.** Deploy the backend first, because the frontend needs its
URL at build time.

1. Push this repository to GitHub.
2. On vercel.com choose Add New, then Project, and import the repository.
3. Set **Root Directory** to `frontend`. The repository root is not the Next.js
   app, and the build fails without this.
4. Leave the framework as Next.js and the build command as detected.
5. Add one environment variable, `NEXT_PUBLIC_API_URL`, set to the Render URL,
   for example `https://siggraph-rag-backend.onrender.com`.
6. Deploy.

Three things about that variable are worth knowing:

- Anything prefixed `NEXT_PUBLIC_` is baked into the browser bundle when the
  project builds. Changing it later has no effect until you redeploy.
- It must start with `https://`. Vercel serves the page over HTTPS, and a
  browser blocks a page from calling a plain `http://` backend as mixed content.
- A trailing slash is harmless; the client strips it.

The page calls `${NEXT_PUBLIC_API_URL}/api/stream` directly rather than going
through the rewrites in `next.config.mjs`, so responses stream straight from
Render to the browser and never pass through Vercel's proxy. That avoids proxy
timeouts. It does mean the request is cross-origin, which works because the
backend echoes the requesting origin in its CORS headers, verified against a
`.vercel.app` origin.

`src/hooks/useRAGWebSocket.ts` exists but nothing imports it. The UI uses
server-sent events. There is no WebSocket traffic to configure.

## Local development

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements-local.txt     # adds fastembed
cp .env.example .env                      # then fill in your keys
python api_server.py
```

`requirements.txt` is the deployment set and installs no model. The local file
adds `fastembed`, which downloads about 1.2 GB of weights on first use and needs
roughly 2.4 GB of RAM. That is why it is kept out of the deployment set.

## Quota

The Cohere trial key allows 1,000 calls per month at 20 per minute. Each query
spends two calls, one to refine the query and one to answer, plus one reranking
call against a separate limit. That is roughly 500 queries per month. Set
`REFINE_QUERY=false` to halve the generation cost if you get close.

Cohere's terms say trial keys are not for production use. That is fine for a
graded assignment demo; a real deployment needs a production key.
