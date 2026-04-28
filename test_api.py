import os
import base64
from openai import OpenAI

client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=os.getenv("NVIDIA_API_KEY", "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")
)

# 1x1 black pixel PNG
b64_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

try:
    completion = client.chat.completions.create(
        model="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What digit is this?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}}
                ]
            }
        ],
        temperature=0.1,
        top_p=0.95,
        max_tokens=256,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}, "reasoning_budget": 16384},
        stream=True
    )
    
    full_text = ""
    for chunk in completion:
        if not chunk.choices:
            continue
        reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning:
            print("REASONING:", reasoning)
        if chunk.choices[0].delta.content is not None:
            full_text += chunk.choices[0].delta.content
            print("CONTENT:", chunk.choices[0].delta.content)
            
    print("FINAL TEXT:", full_text)
except Exception as e:
    print("Error:", e)
