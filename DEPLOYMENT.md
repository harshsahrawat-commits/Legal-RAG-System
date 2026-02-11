# Deployment Guide — Legal RAG System

Production stack: **Railway** (backend) + **Neon** (database) + **Cloudflare Pages** (frontend)

Monthly cost: ~$24/mo ($5 Railway + $19 Neon Launch)

---

## Prerequisites

You need accounts on:
- [GitHub](https://github.com) (your code repo)
- [Railway](https://railway.app) ($5/mo Hobby plan)
- [Neon](https://neon.tech) ($19/mo Launch plan — free tier is only 0.5 GB, your DB is 846 MB)
- [Cloudflare](https://dash.cloudflare.com) (free Pages plan)

You also need these API keys (you should already have them):
- `VOYAGE_API_KEY` — from [Voyage AI](https://dash.voyageai.com/)
- `COHERE_API_KEY` — from [Cohere](https://dashboard.cohere.com/api-keys)
- `NVIDIA_API_KEY` — from [NVIDIA NIM](https://build.nvidia.com/)

---

## Step 1: Push Code to GitHub

If not already done:

```bash
git remote add origin https://github.com/YOUR_USERNAME/Legal-RAG-System.git
git push -u origin main
```

Make sure `.env` is in `.gitignore` (it is). Never push API keys.

---

## Step 2: Set Up Neon Database

### 2A. Create project
1. Go to [Neon Console](https://console.neon.tech)
2. Click **New Project**
3. Project name: `legal-rag`
4. Region: pick closest to your users (e.g., `aws-eu-central-1` for Cyprus/EU)
5. PostgreSQL version: **17**
6. Click **Create Project**

### 2B. Enable pgvector
1. In the Neon SQL Editor, run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2C. Copy connection string
1. Go to **Dashboard** → **Connection Details**
2. Copy the connection string (looks like):
```
postgresql://user:pass@ep-xxxx.eu-central-1.aws.neon.tech/legal_rag?sslmode=require
```
3. Save this — you'll need it for Railway.

### 2D. Migrate your existing data
If you have data in a local/existing database, export and import:
```bash
# Export from current database
pg_dump "$CURRENT_POSTGRES_URL" --no-owner --no-privileges > backup.sql

# Import to Neon
psql "$NEON_POSTGRES_URL" < backup.sql
```

Or use the migration script:
```bash
POSTGRES_URL="$NEON_POSTGRES_URL" python cloud_migrate.py
```

---

## Step 3: Deploy Backend on Railway

### 3A. Create Railway project
1. Go to [Railway Dashboard](https://railway.app/dashboard)
2. Click **New Project** → **Deploy from GitHub Repo**
3. Connect your GitHub account if needed
4. Select your `Legal-RAG-System` repository
5. Railway will auto-detect the `Dockerfile` and `railway.toml`

### 3B. Set environment variables
In Railway, go to your service → **Variables** tab → click **RAW Editor** and paste:

```env
POSTGRES_URL=postgresql://user:pass@ep-xxxx.neon.tech/legal_rag?sslmode=require
VOYAGE_API_KEY=your-voyage-api-key
COHERE_API_KEY=your-cohere-api-key
NVIDIA_API_KEY=your-nvidia-api-key
CORS_ORIGINS=https://your-app.pages.dev,http://localhost:5173
RATE_LIMIT_RPM=60
```

Replace:
- `POSTGRES_URL` with your Neon connection string from Step 2C
- API keys with your real keys
- `your-app.pages.dev` with your Cloudflare Pages domain (you'll get this in Step 4)

### 3C. Generate a public domain
1. In Railway, go to your service → **Settings** → **Networking**
2. Click **Generate Domain**
3. You'll get something like: `legal-rag-production.up.railway.app`
4. Save this URL — the frontend will point to it

### 3D. Verify deployment
Railway will auto-build and deploy. Check:
- **Build Logs**: Should show Docker build completing
- **Deploy Logs**: Should show `Uvicorn running on http://0.0.0.0:PORT`
- **Health check**: Visit `https://YOUR-APP.up.railway.app/api/v1/health`
  - Should return: `{"status": "ok", ...}`

### 3E. Create API key for your tenant
After deployment, use Railway's CLI or run locally:
```bash
# Option 1: Run locally pointing to Neon
POSTGRES_URL="$NEON_POSTGRES_URL" python create_api_key.py

# Option 2: Use Railway CLI
railway run python create_api_key.py
```

---

## Step 4: Deploy Frontend on Cloudflare Pages

### 4A. Connect repository
1. Go to [Cloudflare Dashboard](https://dash.cloudflare.com) → **Workers & Pages**
2. Click **Create** → **Pages** → **Connect to Git**
3. Select your GitHub repo

### 4B. Configure build settings
Set these values:

| Setting | Value |
|---------|-------|
| **Project name** | `legal-rag` (or whatever you want) |
| **Production branch** | `main` |
| **Framework preset** | None |
| **Build command** | `cd frontend && npm install && npm run build` |
| **Build output directory** | `frontend/dist` |
| **Root directory** | `/` (leave default) |

### 4C. Set environment variable
Under **Environment variables**, add:

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | `https://YOUR-APP.up.railway.app` |

Replace with your Railway domain from Step 3C.

### 4D. Deploy
1. Click **Save and Deploy**
2. Cloudflare will build and deploy your frontend
3. You'll get a URL like: `legal-rag.pages.dev`

### 4E. Update Railway CORS
Now go back to Railway → **Variables** and update:
```
CORS_ORIGINS=https://legal-rag.pages.dev,https://YOUR-CUSTOM-DOMAIN.com
```

Redeploy Railway for the CORS change to take effect (Railway auto-redeploys on variable changes).

---

## Step 5: Custom Domain (Optional)

### For Cloudflare Pages (frontend):
1. In Cloudflare Pages → **Custom domains** → **Set up a custom domain**
2. Enter your domain (e.g., `app.yourdomain.com`)
3. If domain is already on Cloudflare, DNS records are auto-created
4. If not, add the CNAME record shown

### For Railway (backend API):
1. In Railway → **Settings** → **Networking** → **Custom Domain**
2. Enter your domain (e.g., `api.yourdomain.com`)
3. Add the CNAME record Railway shows to your DNS

---

## Step 6: Verify Everything Works

### Health check
```bash
curl https://YOUR-RAILWAY-DOMAIN/api/v1/health
# Expected: {"status": "ok", ...}
```

### Frontend
1. Open `https://legal-rag.pages.dev` (or your custom domain)
2. Enter your API key
3. Upload a document or query existing ones
4. Verify answers come back with citations

### API direct test
```bash
curl -X POST https://YOUR-RAILWAY-DOMAIN/api/v1/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"query": "What is this document about?", "top_k": 5}'
```

---

## Ongoing Maintenance

### Automatic deploys
Both Railway and Cloudflare Pages auto-deploy when you push to `main`:
```bash
git push origin main
# → Railway rebuilds backend automatically
# → Cloudflare Pages rebuilds frontend automatically
```

### Database backups
Neon has automatic daily backups on Launch plan. For manual backups:
```bash
pg_dump "$NEON_POSTGRES_URL" --no-owner | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Monitoring
- **Railway**: Built-in logs and metrics at `railway.app/dashboard`
- **Neon**: Query stats and storage monitoring at `console.neon.tech`
- **Cloudflare**: Analytics at `dash.cloudflare.com`

### Scaling
- **More traffic**: Railway auto-scales workers; increase `WORKERS` env var
- **More storage**: Neon Launch gives 10 GB (12x your current 846 MB)
- **More tenants**: Create API keys with `create_api_key.py`, RLS isolates data automatically

---

## Cost Summary

| Service | Plan | Monthly Cost | What You Get |
|---------|------|-------------|--------------|
| Railway | Hobby | $5 | Always-on backend, auto-deploy, logs |
| Neon | Launch | $19 | 10 GB PostgreSQL + pgvector, daily backups, autoscaling |
| Cloudflare Pages | Free | $0 | Unlimited bandwidth, global CDN, auto-deploy |
| **Total** | | **$24/mo** | |

External API costs (usage-based, already paying):
- Voyage AI embeddings
- Cohere reranking
- NVIDIA NIM LLM calls
