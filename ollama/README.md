# Run Ollama over a network securely (SSH)
Install Ollama 
[OLLAMA](https://ollama.com/)

1. ## Bind Ollama to all interfaces
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
   
