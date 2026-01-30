# 🚀 Render Deployment Guide

## Quick Deploy to Render

### Option 1: Deploy from GitHub (Recommended)

1. **Push code to GitHub**:
```bash
git init
git add .
git commit -m "Initial commit for Render deployment"
git branch -M main
git remote add origin https://github.com/yourusername/your-repo.git
git push -u origin main
```

2. **Deploy on Render**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Render will auto-detect `render.yaml` configuration
   - Click "Apply" and "Create Web Service"

### Option 2: Deploy from Local (Blueprint)

1. Install Render CLI:
```bash
npm install -g render
```

2. Login and deploy:
```bash
render login
render blueprint launch
```

## 📋 Configuration Details

Your `render.yaml` includes:
- **Service Type**: Web Service
- **Environment**: Python 3.11
- **Plan**: Free tier
- **Region**: Oregon (change if needed)
- **Health Check**: `/health` endpoint
- **Auto-deploy**: On git push (when connected to GitHub)

## 🔧 Important Notes

### 1. Port Configuration
The app automatically uses Render's `PORT` environment variable. No manual configuration needed.

### 2. Python Version
Specified in `runtime.txt`: Python 3.11.0

### 3. Dependencies
All dependencies in `requirements.txt` will be automatically installed during build.

### 4. OCR Support (Optional)
Render's free tier doesn't include Tesseract OCR by default. For PDFs with selectable text, this works fine. For scanned PDFs requiring OCR:

Add to `render.yaml` under the service:
```yaml
buildCommand: |
  apt-get update
  apt-get install -y tesseract-ocr poppler-utils
  pip install -r requirements.txt
```

Note: This may increase build time and might not be available on free tier.

## 🌐 After Deployment

Once deployed, you'll get a URL like:
```
https://ai-battle-rag.onrender.com
```

### Test Your Deployment:
1. Health check: `https://your-app.onrender.com/health`
2. Frontend: `https://your-app.onrender.com/`
3. API endpoint: `https://your-app.onrender.com/aibattle`

## ⚡ Performance Notes

### Free Tier Limitations:
- Service spins down after 15 minutes of inactivity
- First request after spin-down takes 30-60 seconds (cold start)
- 512 MB RAM limit
- Shared CPU

### Upgrade for Better Performance:
- Starter ($7/month): Always on, no spin-down
- Standard ($25/month): More resources, faster processing

## 🔒 Environment Variables (Optional)

If you need to add environment variables later:
1. Go to your service dashboard on Render
2. Navigate to "Environment" tab
3. Add key-value pairs

Currently, no API keys are required (100% offline system).

## 📊 Monitoring

Render provides:
- Real-time logs in dashboard
- Automatic health checks via `/health` endpoint
- Metrics and analytics (on paid plans)

## 🐛 Troubleshooting

### Build Fails
- Check logs in Render dashboard
- Verify `requirements.txt` dependencies
- Ensure Python 3.11 compatibility

### Service Unhealthy
- Check `/health` endpoint response
- Review application logs
- Verify port binding (should auto-detect PORT)

### Slow Performance
- Free tier has resource limits
- Consider upgrading to Starter plan
- Optimize `response_preferences.json` settings

### OCR Not Working
- Add Tesseract to build command (see above)
- Or ensure PDFs have selectable text

## 🔄 Update Deployment

### From GitHub:
```bash
git add .
git commit -m "Update"
git push
```
Render auto-deploys on push.

### Manual Redeploy:
Go to Render dashboard → Click "Manual Deploy" → "Deploy latest commit"

## 📞 Support

- Render Docs: https://render.com/docs
- Render Community: https://community.render.com
- Check logs in dashboard for error details

## ✅ Deployment Checklist

- [x] `render.yaml` configured
- [x] `requirements.txt` with all dependencies
- [x] `runtime.txt` specifying Python 3.11
- [x] Port configuration using environment variable
- [x] Health check endpoint at `/health`
- [x] CORS enabled for frontend
- [ ] Push code to GitHub
- [ ] Create Render account
- [ ] Connect repository to Render
- [ ] Deploy and test!
