# Setup Guide: Automated Dataset Diagnostics and Repair Framework (ADIE)

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Running the Application](#running-the-application)
5. [Testing the Installation](#testing-the-installation)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Configuration](#advanced-configuration)

## System Requirements

### Minimum Requirements
- **Operating System**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 4GB minimum (8GB recommended for large datasets)
- **Storage**: 2GB free space (additional space for datasets)
- **Processor**: Dual-core CPU (Quad-core recommended)

### Recommended Requirements
- **Operating System**: Windows 11 or Ubuntu 20.04+
- **Python**: 3.9 or 3.10
- **RAM**: 16GB or more
- **Storage**: 10GB free space
- **Processor**: Quad-core CPU or better

### Software Dependencies
- Git (for cloning and version control)
- Web browser (Chrome, Firefox, Safari, or Edge)
- Text editor or IDE (VS Code, PyCharm, etc.)

## Installation

### Step 1: Clone or Download the Project

#### Option A: Clone from Git Repository
```bash
git clone https://github.com/your-username/ai-project.git
cd ai-project
```

#### Option B: Download ZIP File
1. Download the project ZIP file
2. Extract to your desired location
3. Navigate to the project directory in terminal/command prompt

### Step 2: Create Virtual Environment

#### Windows
```cmd
# Create virtual environment
python -m venv adie_env

# Activate virtual environment
adie_env\Scripts\activate
```

#### macOS/Linux
```bash
# Create virtual environment
python3 -m venv adie_env

# Activate virtual environment
source adie_env/bin/activate
```

### Step 3: Install Dependencies

#### Install Basic Requirements
```bash
pip install -r requirements.txt
```

#### Install Advanced Dependencies (Optional)
```bash
pip install -r requirements-advanced.txt
```

#### Manual Installation (if requirements.txt fails)
```bash
# Core dependencies
pip install flask==2.3.3
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install imbalanced-learn==0.11.0

# Web interface dependencies
pip install jinja2==3.1.2
pip install werkzeug==2.3.7
pip install click==8.1.7

# Additional utilities
pip install seaborn==0.12.2
pip install matplotlib==3.7.2
pip install plotly==5.15.0
```

### Step 4: Verify Installation

```bash
# Test Python packages
python -c "import flask, pandas, numpy, sklearn; print('All packages imported successfully!')"

# Check versions
python -c "import flask, pandas, numpy, sklearn; print(f'Flask: {flask.__version__}'); print(f'Pandas: {pandas.__version__}'); print(f'NumPy: {numpy.__version__}'); print(f'Scikit-learn: {sklearn.__version__}')"
```

## Configuration

### Step 1: Create Required Directories

```bash
mkdir uploads
mkdir static/reports
mkdir logs
```

### Step 2: Environment Variables

Create a `.env` file in the project root:

```env
# Flask Configuration
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here-change-this-in-production

# File Upload Configuration
MAX_CONTENT_LENGTH=104857600  # 100MB in bytes
UPLOAD_FOLDER=uploads

# Database Configuration (if using database)
DATABASE_URL=sqlite:///adie.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/adie.log

# Performance Configuration
MAX_DATASET_SIZE=500000
CHUNK_SIZE=10000
```

### Step 3: Application Configuration

Edit `app.py` if needed:

```python
# Configuration settings
app.config.update(
    SECRET_KEY=os.environ.get('SECRET_KEY', 'dev-secret-key'),
    MAX_CONTENT_LENGTH=int(os.environ.get('MAX_CONTENT_LENGTH', 104857600)),
    UPLOAD_FOLDER=os.environ.get('UPLOAD_FOLDER', 'uploads'),
    DEBUG=os.environ.get('FLASK_ENV', 'development') == 'development'
)
```

### Step 4: File Permissions

#### Windows
```cmd
# Grant write permissions to uploads folder
icacls uploads /grant Everyone:F
```

#### macOS/Linux
```bash
# Set appropriate permissions
chmod 755 uploads
chmod 644 *.py
chmod 644 requirements.txt
chmod 644 setup.md
```

## Running the Application

### Method 1: Development Mode

#### Start the Application
```bash
# Make sure virtual environment is activated
python app.py
```

#### Expected Output
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
Press CTRL+C to quit
 * Restarting with stat
 * Debugger is active!
 * Debugger PIN: 123-456-789
```

### Method 2: Production Mode

#### Using Gunicorn (Linux/macOS)
```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

#### Using Waitress (Windows)
```bash
# Install waitress
pip install waitress

# Run with waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

### Method 3: Docker (Optional)

#### Create Dockerfile
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

#### Build and Run
```bash
# Build Docker image
docker build -t adie-framework .

# Run container
docker run -p 5000:5000 -v $(pwd)/uploads:/app/uploads adie-framework
```

## Accessing the Application

### Web Interface
1. Open your web browser
2. Navigate to `http://localhost:5000`
3. You should see the ADIE framework homepage

### API Endpoints
- **Homepage**: `GET /`
- **Upload**: `POST /upload`
- **Analyze**: `POST /analyze`
- **Repair**: `POST /repair`
- **Report**: `GET /report`
- **Download**: `GET /download/<file_type>`

## Testing the Installation

### Step 1: Upload Test Dataset

1. Navigate to `http://localhost:5000`
2. Click "Choose File" and select a CSV file
3. Click "Upload and Analyze"
4. Wait for analysis to complete

### Step 2: Verify Analysis Results

Check that the following are displayed:
- Dataset information (rows, columns, data types)
- Detected issues with severity ratings
- Repair recommendations
- Performance improvement estimates

### Step 3: Test Repair Process

1. Review detected issues
2. Click "Apply Repairs"
3. Wait for processing to complete
4. Review performance comparison results

### Step 4: Download Results

1. Click "Download Repaired Dataset"
2. Click "Download Report"
3. Verify files are downloaded correctly

### Step 5: Run Unit Tests

```bash
# Run all tests
python -m pytest test_*.py -v

# Run specific test
python -m pytest test_fixes.py -v

# Run with coverage
python -m pytest --cov=. --cov-report=html
```

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: Module Import Errors
**Error**: `ModuleNotFoundError: No module named 'flask'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Windows
adie_env\Scripts\activate

# macOS/Linux
source adie_env/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### Issue 2: Port Already in Use
**Error**: `OSError: [Errno 98] Address already in use`

**Solution**:
```bash
# Find process using port 5000
# Windows
netstat -ano | findstr :5000

# macOS/Linux
lsof -i :5000

# Kill the process or use different port
python app.py --port 5001
```

#### Issue 3: File Upload Errors
**Error**: `File size too large` or `Permission denied`

**Solution**:
```bash
# Check file size limit in app.py
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Check folder permissions
# Windows
icacls uploads /grant Everyone:F

# macOS/Linux
chmod 755 uploads
```

#### Issue 4: Memory Errors with Large Datasets
**Error**: `MemoryError` or application crashes

**Solution**:
```bash
# Increase chunk size in configuration
CHUNK_SIZE=5000

# Or use sampling for analysis
python app.py --sample-size 10000
```

#### Issue 5: Slow Performance
**Symptoms**: Long processing times, unresponsive interface

**Solution**:
```bash
# Enable debug mode to identify bottlenecks
export FLASK_ENV=development

# Use sampling for large datasets
# Modify app.py to enable sampling
df_sample = df.sample(n=min(10000, len(df)))
```

### Debug Mode

Enable debug mode for detailed error messages:

```bash
# Set environment variable
export FLASK_ENV=development

# Or modify app.py
app.run(debug=True)
```

### Logging

Check application logs for errors:

```bash
# View log file
tail -f logs/adie.log

# Or enable console logging
python app.py --log-level DEBUG
```

## Advanced Configuration

### Database Integration

#### SQLite Setup
```python
# Add to app.py
import sqlite3
from contextlib import closing

def init_db():
    with closing(sqlite3.connect('adie.db')) as db:
        with app.open_resource('schema.sql', mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
```

#### PostgreSQL Setup
```bash
# Install PostgreSQL adapter
pip install psycopg2-binary

# Update environment variables
DATABASE_URL=postgresql://username:password@localhost/adie_db
```

### Performance Optimization

#### Caching Configuration
```python
# Add to app.py
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@cache.memoize(timeout=300)
def expensive_analysis(df):
    # Your analysis code here
    pass
```

#### Background Processing
```python
# Use Celery for long-running tasks
from celery import Celery

celery = Celery('adie', broker='redis://localhost:6379/0')

@celery.task
def background_analysis(filepath):
    # Your analysis code here
    pass
```

### Security Configuration

#### HTTPS Setup
```python
# Add to app.py
from flask_sslify import SSLify

if app.config['DEBUG'] is False:
    SSLify(app)
```

#### Authentication
```python
# Add to app.py
from flask_login import LoginManager

login_manager = LoginManager()
login_manager.init_app(app)
```

### Monitoring and Analytics

#### Health Check Endpoint
```python
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now()}
```

#### Metrics Collection
```python
# Add Prometheus metrics
from prometheus_client import Counter, generate_latest

REQUEST_COUNT = Counter('requests_total', 'Total requests', ['method', 'endpoint'])

@app.before_request
def before_request():
    REQUEST_COUNT.labels(method=request.method, endpoint=request.endpoint).inc()
```

## Production Deployment

### Heroku Deployment

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### AWS Deployment

1. Use AWS Elastic Beanstalk
2. Configure environment variables
3. Set up load balancer
4. Enable auto-scaling

### Docker Swarm/Kubernetes

1. Create `docker-compose.yml`
2. Deploy to cluster
3. Configure persistent storage
4. Set up monitoring

## Support and Maintenance

### Regular Maintenance Tasks

1. **Update Dependencies**:
```bash
pip list --outdated
pip install --upgrade package-name
```

2. **Clean Uploads**:
```bash
# Clean files older than 30 days
find uploads -mtime +30 -delete
```

3. **Backup Database**:
```bash
# SQLite
cp adie.db backup/adie_$(date +%Y%m%d).db

# PostgreSQL
pg_dump adie_db > backup/adie_$(date +%Y%m%d).sql
```

### Getting Help

1. **Check Logs**: `tail -f logs/adie.log`
2. **Review Documentation**: Check inline code comments
3. **Community Support**: GitHub issues and discussions
4. **Email Support**: support@your-domain.com

### Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

---

## Quick Start Summary

For experienced users, here's the fastest way to get started:

```bash
# 1. Setup environment
git clone <repository-url>
cd ai-project
python -m venv adie_env
source adie_env/bin/activate  # Windows: adie_env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create directories
mkdir uploads logs

# 4. Run application
python app.py

# 5. Access application
# Open http://localhost:5000 in browser
```

That's it! Your ADIE framework should now be running and ready to use.
