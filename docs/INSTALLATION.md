# Installation & Setup Guide - Codesearch

This guide covers installation, configuration, and deployment of Codesearch for different scenarios.

## System Requirements

### Minimum Requirements
- Python 3.9 or later
- 4 GB RAM
- 2 GB free disk space (for database)
- macOS, Linux, or Windows

### Recommended Requirements
- Python 3.10 or later
- 8 GB RAM
- 10+ GB free SSD space
- Linux or macOS (best support)

### Large Project Requirements (50K+ entities)
- Python 3.11+
- 16 GB RAM
- 100+ GB SSD space
- Dedicated machine

## Installation Methods

### Method 1: pip (Recommended)

**From PyPI (when released):**
```bash
pip install codesearch
```

**From GitHub (current development):**
```bash
pip install git+https://github.com/jpequegn/codesearch.git
```

**From Local Repository:**
```bash
# Clone the repository
git clone https://github.com/jpequegn/codesearch.git
cd codesearch

# Install in development mode
pip install -e .
```

### Method 2: From Source

```bash
# Clone repository
git clone https://github.com/jpequegn/codesearch.git
cd codesearch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Method 3: Docker (Future)

```bash
# Build image
docker build -t codesearch:latest .

# Run container
docker run -v /path/to/repos:/repos -v ~/.codesearch:/root/.codesearch \
  codesearch:latest pattern "validation"
```

## Development Setup

### Prerequisites
- Git
- Python 3.9+
- pip and virtualenv

### Initial Setup

```bash
# 1. Clone repository
git clone https://github.com/jpequegn/codesearch.git
cd codesearch

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install with dev dependencies
pip install -e ".[dev]"

# 5. Verify installation
codesearch --version
pytest --version
black --version
```

### IDE Setup

**Visual Studio Code:**
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": false,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "ms-python.python",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "ms-python.python"
  }
}
```

**PyCharm:**
1. File → Settings → Project → Python Interpreter
2. Add Interpreter → Add Local Interpreter → Existing Environment
3. Select `venv/bin/python`
4. Enable "Reformat Code" on save in settings

## Configuration

### Environment Variables

Create `.env` file in project root (see `.env.example` for template):

```bash
# Copy template
cp .env.example .env

# Edit with your settings
nano .env  # or your editor
```

**Common variables:**
```bash
# Database location
export CODESEARCH_DB_PATH=~/.codesearch

# Default language
export CODESEARCH_LANGUAGE=python

# Output format
export CODESEARCH_OUTPUT_FORMAT=table
```

### Configuration Files

**Python dotenv (recommended):**
```bash
# Create .env file in project root
CODESEARCH_DB_PATH=~/.codesearch
CODESEARCH_LANGUAGE=python
```

**System environment (bash/zsh):**
```bash
# Add to ~/.bashrc or ~/.zshrc
export CODESEARCH_DB_PATH=~/.codesearch
export CODESEARCH_LANGUAGE=python
```

**YAML configuration (future):**
```bash
# Copy config template
cp config/config.yaml.example ~/.codesearch/config.yaml

# Edit configuration
nano ~/.codesearch/config.yaml
```

## Post-Installation

### Verify Installation

```bash
# Check version
codesearch --version

# Display help
codesearch --help

# List all commands
codesearch --help
```

### Initialize Database

```bash
# Index your first repository
codesearch index /path/to/repo

# Verify indexing worked
codesearch statistics
```

### Run Quick Test

```bash
# Search for something
codesearch pattern "function that validates"

# If you indexed your repo
codesearch find-similar main
```

## Troubleshooting Installation

### Issue: "command not found: codesearch"

**Solution:**
1. Verify installation: `pip list | grep codesearch`
2. Check Python path: `which python`
3. Reinstall: `pip install --force-reinstall -e .`
4. Check virtual environment is activated

### Issue: "No module named 'transformers'"

**Solution:**
```bash
# Install dependencies separately
pip install transformers torch

# Reinstall codesearch
pip install -e .
```

### Issue: torch installation fails

**Solution:**
```bash
# For macOS M1/M2/M3:
conda install pytorch::pytorch torchvision torchaudio -c pytorch

# For Windows/Linux:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: Database permission denied

**Solution:**
```bash
# Create directory with proper permissions
mkdir -p ~/.codesearch
chmod 755 ~/.codesearch

# Or use different location
export CODESEARCH_DB_PATH=/tmp/codesearch
mkdir -p $CODESEARCH_DB_PATH
```

## Upgrade

### From Previous Version

```bash
# Upgrade to latest
pip install --upgrade codesearch

# Or from source
cd codesearch
git pull origin main
pip install --upgrade -e .

# Verify upgrade
codesearch --version
```

### Database Migration

```bash
# Backup old database
cp -r ~/.codesearch ~/.codesearch.backup

# Rebuild database
rm -rf ~/.codesearch
codesearch index /path/to/repo
```

## Uninstall

```bash
# Remove package
pip uninstall codesearch

# Remove database (optional)
rm -rf ~/.codesearch

# Remove cache
rm -rf ~/.codesearch/models
rm -rf ~/.codesearch/cache
```

## Platform-Specific Guides

### macOS

**Installation:**
```bash
# Using Homebrew (when available)
brew install codesearch

# Or using pip (current)
pip install codesearch
```

**Apple Silicon (M1/M2/M3):**
```bash
# Use conda for better compatibility
conda create -n codesearch python=3.11
conda activate codesearch
conda install pytorch::pytorch -c pytorch
pip install codesearch
```

### Linux

**Ubuntu/Debian:**
```bash
# Install Python development files
sudo apt-get install python3.11 python3.11-venv python3.11-dev

# Install codesearch
python3.11 -m pip install codesearch
```

**Fedora/RHEL:**
```bash
# Install Python development files
sudo dnf install python3.11 python3.11-devel

# Install codesearch
python3.11 -m pip install codesearch
```

### Windows

**PowerShell:**
```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install codesearch
pip install codesearch

# Verify
codesearch --version
```

**Windows Subsystem for Linux (WSL):**
```bash
# Use Linux installation instructions
# WSL provides native Python environment
pip install codesearch
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install codesearch
COPY . .
RUN pip install -e .

# Create volumes
VOLUME ["/data/repos", "/data/db"]

# Set environment
ENV CODESEARCH_DB_PATH=/data/db

# Run as non-root
RUN useradd -m -u 1000 codesearch
USER codesearch

ENTRYPOINT ["codesearch"]
CMD ["--help"]
```

**Build and run:**
```bash
# Build
docker build -t codesearch:latest .

# Run
docker run -v /path/to/repos:/data/repos \
  -v ~/.codesearch:/data/db \
  codesearch:latest pattern "validation"
```

### Kubernetes Deployment (Future)

Configuration for running Codesearch in Kubernetes environments.

### Systemd Service (Linux)

Create `/etc/systemd/system/codesearch-indexer.service`:

```ini
[Unit]
Description=Codesearch Indexing Service
After=network.target

[Service]
Type=oneshot
User=codesearch
WorkingDirectory=/opt/codesearch
Environment="CODESEARCH_DB_PATH=/var/lib/codesearch"
ExecStart=/usr/local/bin/codesearch index /opt/repos

[Install]
WantedBy=multi-user.target
```

Enable and run:
```bash
sudo systemctl enable codesearch-indexer
sudo systemctl start codesearch-indexer
```

## Health Checks

### Verify Installation

```bash
# 1. Check version
codesearch --version

# 2. Check database
ls -lh ~/.codesearch/

# 3. Test search
codesearch pattern "test" --limit 1

# 4. Check config
cat .env
```

### Performance Testing

```bash
# Time a search operation
time codesearch pattern "validation" --limit 10

# Check memory usage
ps aux | grep codesearch

# Monitor database size
du -sh ~/.codesearch/
```

## Getting Help

- **Documentation**: See [docs/](../docs/)
- **Troubleshooting**: See [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **Issues**: https://github.com/jpequegn/codesearch/issues
- **Discussions**: https://github.com/jpequegn/codesearch/discussions

## Next Steps

1. **Configure**: Copy `.env.example` to `.env` and customize
2. **Index**: Run `codesearch index /path/to/repo`
3. **Search**: Try `codesearch pattern "your query"`
4. **Develop**: See [CONTRIBUTING.md](../CONTRIBUTING.md) for development guide

---

**Ready to start?** Follow the [Quick Start](../README.md#quick-start) guide!
