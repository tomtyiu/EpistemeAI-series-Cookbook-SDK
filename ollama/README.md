# Run Ollama over a network securely (SSH)
Install Ollama 
- [OLLAMA](https://ollama.com/)

Recommended model: 
```
ollama run jimfeedback/Reasoning-Medical-27B-i1-GGUF
```
or

```
ollama run hf.co/mradermacher/Reasoning-Medical-27B-i1-GGUF:IQ3_M
```

## Step 1. Bind Ollama to all interfaces
```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

If Ollama runs as a systemd service, add the environment variable to its service override:
```
sudo systemctl edit ollama
```

Add the following in the editor that opens:
```
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

Save, then reload:
```
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

2. ## Make it permanent on  Windows
Set a system environment variable:

Open System Properties → Environment Variables
Under System Variables, click New
Variable name: OLLAMA_HOST, value: 0.0.0.0
Restart the Ollama application

3. ## Make it permanent on Mac
Quit the Ollama app, then launch it with the environment variable set via launchd or by starting from the terminal:Development Tools
```
OLLAMA_HOST=0.0.0.0 ollama serve
```
To persist it, create a launchd plist at ~/Library/LaunchAgents/ollama.plist and include the OLLAMA_HOST environment key.
   

## Step 2: Open the Firewall
On most systems, a firewall will block incoming connections on port 11434 by default. You need to allow it explicitly.

Linux (ufw)
# Allow from your LAN only (recommended)
```
sudo ufw allow from 192.168.1.0/24 to any port 11434
```

# Or allow from everywhere (not recommended for production)
```
sudo ufw allow 11434/tcp
```

# Step 3: Test the Connection
From another machine on the same network, replace 192.168.1.50 with your Ollama server’s IP address:

```
curl http://192.168.1.50:11434/api/tags
```
You should see a JSON response listing your pulled models. If you get a connection refused or timeout, check the firewall rules and that Ollama is bound to the right interface.
