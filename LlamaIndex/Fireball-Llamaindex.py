# %pip install llama-index-llms-text-generation-inference
import os
from typing import List, Optional

from llama_index.llms.text_generation_inference import (
    TextGenerationInference,
)

URL = "your_tgi_endpoint"
model = TextGenerationInference(
    model_url=URL, token=False
)  # set token to False in case of public endpoint

completion_response = model.complete("To infinity, and")
print(completion_response)
