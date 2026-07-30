# Ollama

# Agent Medical Reasoning Chatbot with Ollama

A command-line medical reasoning chatbot powered by a locally running GGUF model through Ollama.

The application:

- Accepts natural-language questions from the terminal.
- Sends each question to the configured medical reasoning model.
- Streams the model response as it is generated.
- Continues accepting questions until the user types `bye`.
- Handles connection, model, and runtime errors without crashing the entire program.

> **Medical safety notice:** This project is intended for research, education, and experimentation. It is not a substitute for diagnosis, treatment, emergency care, or advice from a qualified healthcare professional. Do not enter private patient information or other sensitive health data unless your environment has been reviewed for the applicable privacy and security requirements.

---

## Model

The code uses the following model by default:

```text
hf.co/mradermacher/Reasoning-Medical0.1-E4B-sft-GGUF:Q8_0
```

This is a Q8_0-quantized GGUF model loaded through Ollama.

Because Q8_0 models can require substantial memory and storage, confirm that the computer has enough RAM, available disk space, and—when applicable—GPU memory.

---

## Requirements

Before running the chatbot, install:

- Python 3.8 or later
- Ollama
- The Ollama Python package
- The configured GGUF model

Ollama must be running while the Python application is being used.

---

## Project Structure

A minimal project can use the following layout:

```text
agent-medical-chatbot/
├── medical_chatbot.py
└── README.md
```

Save the supplied Python code as:

```text
medical_chatbot.py
```

---

## 1. Install Ollama

Download and install Ollama for your operating system from the official Ollama website:

- [Ollama](https://ollama.com/)
- [Ollama documentation](https://docs.ollama.com/)

After installation, confirm that Ollama is available:

```bash
ollama --version
```

If Ollama is not already running, start it:

```bash
ollama serve
```

Depending on the operating system, the Ollama desktop application may start the service automatically.

---

## 2. Create a Python Environment

Creating a virtual environment is recommended.

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows Command Prompt

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install the Python Dependency

Install the official Ollama Python client:

```bash
pip install --upgrade ollama
```

Optional `requirements.txt`:

```text
ollama
```

Install from that file with:

```bash
pip install -r requirements.txt
```

---

## 4. Download the Model

Pull the configured Hugging Face GGUF model through Ollama:

```bash
ollama pull hf.co/mradermacher/Reasoning-Medical0.1-E4B-sft-GGUF:Q8_0
```

You can also test the model directly before running the Python application:

```bash
ollama run hf.co/mradermacher/Reasoning-Medical0.1-E4B-sft-GGUF:Q8_0
```

Enter a test prompt, then use the appropriate terminal command to exit the interactive Ollama session.

To verify that the model is installed:

```bash
ollama list
```

---

## 5. Run the Chatbot

From the project directory, run:

### Windows

```powershell
python medical_chatbot.py
```

### macOS or Linux

```bash
python3 medical_chatbot.py
```

The program displays:

```text
=======================================================================
Welcome to your own Agent Medical Reasoning 27B Code Chatbot
=======================================================================
Type 'bye' to exit
```

Enter a question after the prompt:

```text
USER>> Explain the difference between type 1 and type 2 diabetes.
```

The response is streamed to the terminal as the model generates it.

To close the chatbot, enter:

```text
bye
```

---

## How the Code Works

### Importing Ollama

```python
import ollama
```

This imports the Python client used to communicate with the local Ollama service.

### Chatbot Function

```python
def chatbot(input_user):
```

The function receives a single user question and starts a new chat request.

### System Instruction

The system message defines the assistant's intended behavior, including:

- Medical-assistant behavior
- Structured problem decomposition
- Careful, respectful, and truthful responses
- Safety constraints
- Resistance to prompt injection and SQL injection

A system prompt can guide behavior, but it cannot guarantee security. Applications that handle untrusted data should enforce security controls in code rather than relying only on model instructions.

### Model Request

```python
stream = ollama.chat(
    model='hf.co/mradermacher/Reasoning-Medical0.1-E4B-sft-GGUF:Q8_0',
    messages=[...],
    stream=True,
)
```

The request:

1. Selects the configured model.
2. sends the system and user messages.
3. enables streaming with `stream=True`.

### Streaming Output

```python
for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

Each generated chunk is printed immediately instead of waiting for the entire response.

### Error Handling

```python
except Exception as e:
    print(f"Error occurred during chat: {str(e)}")
```

This catches common failures such as:

- Ollama is not running.
- The model is unavailable.
- The model name is incorrect.
- The machine does not have enough memory.
- The connection to Ollama fails.

### Interactive Loop

```python
while query.lower() != "bye":
    query = input("\nUSER>> ")
    chatbot(query)
```

The loop repeatedly accepts questions and sends them to the model.

---

## Important Current Behavior

### Each Question Starts a New Conversation

The program does not preserve earlier messages. Every call to `chatbot()` sends only the current question and the fixed system messages.

For example, the model will not reliably understand a follow-up such as:

```text
USER>> What about its treatment?
```

unless the subject is included again.

To support multi-turn memory, store prior user and assistant messages in a shared `messages` list and send the complete list with each request.

### The `bye` Message Is Still Sent to the Model

With the current loop, `chatbot("bye")` is called before the loop ends. The model may therefore respond once more after the user types `bye`.

A cleaner exit pattern is:

```python
while True:
    query = input("\nUSER>> ").strip()

    if query.lower() == "bye":
        print("Goodbye.")
        break

    chatbot(query)
```

### Brave Search Is Not Actually Called

This line:

```python
{'role': 'user', 'content': "brave_search.call(query='{input_user}')"},
```

only sends literal text to the model. It does **not** execute Brave Search, and `{input_user}` is not interpolated because the string is not an f-string.

Changing it to:

```python
{'role': 'user', 'content': f"brave_search.call(query='{input_user}')"},
```

would insert the question into the text, but it would still not call a search service.

Real web search requires:

1. A Brave Search API account and API key.
2. Python code that sends the query to the API.
3. Validation and sanitization of the returned data.
4. Passing the search results to the model as context.
5. Clear source attribution in the final answer.

Until that integration is implemented, remove the simulated tool-call message to avoid making the model believe that a search occurred.

### The Application Does Not Implement Ollama Tool Calling

The system prompt mentions tool calling, but the `ollama.chat()` request does not define a `tools` parameter or execute returned tool calls.

Tool use requires application-side logic that:

1. Defines permitted functions.
2. sends those definitions to the model.
3. detects a returned tool call.
4. validates its arguments.
5. executes the approved function.
6. sends the function result back to the model.
7. asks the model to produce the final answer.

---

## Configuration

### Change the Model

Edit the `model` value:

```python
model='hf.co/mradermacher/Reasoning-Medical0.1-E4B-sft-GGUF:Q8_0'
```

Replace it with another model already available to Ollama.

After changing the model, pull it before running the application:

```bash
ollama pull MODEL_NAME
```

### Change the System Prompt

Edit the first message in the `messages` list:

```python
{'role': 'system', 'content': 'Your revised system instruction'}
```

Keep instructions clear and avoid contradictory requirements.

### Disable Streaming

Change:

```python
stream=True
```

to:

```python
stream=False
```

When streaming is disabled, the response must be handled as one object rather than iterated over as chunks.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'ollama'`

Install the package in the active Python environment:

```bash
python -m pip install --upgrade ollama
```

Confirm that `python` and `pip` refer to the same environment:

```bash
python -m pip show ollama
```

### Cannot Connect to Ollama

Confirm that Ollama is running:

```bash
ollama serve
```

Then test the service by listing installed models:

```bash
ollama list
```

### Model Not Found

Pull the exact model name used in the code:

```bash
ollama pull hf.co/mradermacher/Reasoning-Medical0.1-E4B-sft-GGUF:Q8_0
```

Model identifiers are case-sensitive in some environments, so copy the name exactly.

### The Model Loads Very Slowly

The first request may take longer because Ollama must load the model into memory. A Q8_0 GGUF model can also be computationally demanding.

Possible improvements include:

- Using a smaller or more heavily quantized model.
- Closing memory-intensive applications.
- Confirming that hardware acceleration is active.
- Reducing the model context size through an Ollama `Modelfile` when appropriate.

### Out-of-Memory Error

Use a smaller quantization or model size, or run the application on a machine with more available memory.

### Blank or Partial Output

Confirm that returned chunks contain the expected message structure. Newer versions of the client may also support attribute-style access:

```python
print(chunk.message.content, end='', flush=True)
```

---

## Security Considerations

The chatbot accepts arbitrary user input. For local experimentation, the current design is simple, but production use requires stronger controls.

Recommended controls include:

- Do not execute text generated by the model as shell commands, SQL, Python, or API calls.
- Use an allowlist for every callable tool.
- Validate tool arguments with a strict schema.
- Apply request timeouts and output-size limits.
- Keep API keys in environment variables, not source code.
- Treat web content and retrieved documents as untrusted input.
- Log tool activity without storing private medical data.
- Require human review for clinically significant decisions.
- Do not expose the Ollama service directly to the public internet without authentication and network protections.

---

## Privacy

Ollama can run models locally, but local execution alone does not guarantee privacy.

Before processing sensitive information:

- Review where prompts, logs, terminal history, and model outputs are stored.
- Disable unnecessary logging.
- Restrict access to the host machine.
- Encrypt sensitive storage.
- Avoid entering names, dates of birth, medical-record numbers, or other identifying information.
- Review applicable organizational policies and legal requirements.

---

## Suggested Production Improvements

The current script is suitable as a basic command-line demonstration. A more complete application could add:

- Multi-turn conversation memory
- A clean exit before sending `bye`
- Configurable model names through environment variables
- Structured logging
- Token and context limits
- Real Brave Search integration
- Verifiable citations for web-supported answers
- Ollama-native function/tool definitions
- Input and output moderation
- Retrieval-augmented generation from trusted medical sources
- Automated tests
- A web interface using Gradio, Streamlit, or FastAPI
- Explicit emergency guidance for urgent medical prompts
- A privacy notice and user consent flow

---

## Example Session

```text
=======================================================================
Welcome to your own Agent Medical Reasoning 27B Code Chatbot
=======================================================================
Type 'bye' to exit

USER>> What is hypertension?

Hypertension is persistently elevated blood pressure. It can increase the risk
of cardiovascular, kidney, and other health problems. Diagnosis should be based
on properly measured readings and evaluation by a qualified healthcare
professional.

USER>> bye
```

Actual output depends on the model, Ollama version, model parameters, and hardware.

---

## License

No license is included automatically with this example.

Before distributing the project, review:

- The license for this source code
- The model's license
- The quantizer or model distributor's terms
- The source model's acceptable-use requirements
- Any third-party API terms used by future search integrations

Add an appropriate `LICENSE` file before publishing or redistributing the project.

---


## Run Ollama over a network securely (SSH)
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


## References

- [Ollama Python library](https://github.com/ollama/ollama-python)
- [Ollama documentation](https://docs.ollama.com/)
- [Importing GGUF models into Ollama](https://docs.ollama.com/import)

