import os
import requests
import json

PROMPT_PATH = os.path.join(os.path.dirname(__file__), 'server_summary.prompt')
MODEL_URL = "http://localhost:8000/v1/chat/completions"
MODEL_NAME = "/PATH/TO/Qwen2.5-72B-Instruct"
HEADERS = {"Content-Type": "application/json"}

# path for ReadMe files (download from the official MCP repo)
READMES_ROOTS = [
    ("community", "readmes/community"),
    ("official", "readmes/official"),
    ("reference", "readmes/reference"),
]
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), 'output')

# Read prompt template
with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
    prompt_template = f.read()

def extract_info_for_readme(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read()
    prompt = prompt_template.replace('<|README_FILE_PLACEHOLDER|>', readme_content)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    try:
        response = requests.post(MODEL_URL, headers=HEADERS, json=payload, timeout=120)
        if response.status_code == 200:
            result = response.json()
            text = result["choices"][0]["message"]["content"]
            if isinstance(text, list):
                for item in text:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text", "")
                        break
            return text
        else:
            print(f"[ERROR] {readme_path}: {response.status_code} {response.text}")
            return None
    except Exception as e:
        print(f"[EXCEPTION] {readme_path}: {e}")
        return None

def main():
    for group, folder in READMES_ROOTS:
        output_dir = os.path.join(OUTPUT_ROOT, group)
        os.makedirs(output_dir, exist_ok=True)
        for fname in os.listdir(folder):
            if not fname.endswith('.md'):
                continue
            readme_path = os.path.join(folder, fname)
            print(f"Processing {readme_path} ...")
            result = extract_info_for_readme(readme_path)
            if result:
                # Save output as JSON file
                out_path = os.path.join(output_dir, fname.replace('.md', '.json'))
                try:
                    # Try to pretty-format if valid JSON, else save as raw text
                    parsed = json.loads(result)
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
                except Exception:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        f.write(result)

if __name__ == "__main__":
    main() 