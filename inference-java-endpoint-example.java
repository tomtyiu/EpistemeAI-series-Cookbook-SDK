// If necessary, install the openai JavaScript library by running 
// npm install --save openai

import OpenAI from "openai";

const openai = new OpenAI({
    "baseURL": "https://kpu92bguf2t8bjdg.us-east-1.aws.endpoints.huggingface.cloud/v1/",
    "apiKey": "hf_XXXXX"
});

const stream = await openai.chat.completions.create({
    "model": "tgi",
    "messages": [
        {
            "role": "user",
            "content": "What is deep learning?"
        }
    ],
    "max_tokens": 150,
    "stream": true
});

for await (const chunk of stream) {
	process.stdout.write(chunk.choices[0]?.delta?.content || '');
}