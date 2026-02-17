import requests
import json
import os
import re
import sys

# -------------------------
# Configuration
# -------------------------
APP_ID = "9J7EAHLQ5U"
API_URL = "http://api.wolframalpha.com/v2/query"
OUTPUT_DIR = "wolfram_data"  # folder to store JSON files
IMAGES_SUBDIR = "images"  # subfolder for downloaded images

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _safe_folder_name(query):
    """Make a filesystem-safe folder name from the query."""
    s = query.replace(" ", "_")
    s = re.sub(r'[<>:"/\\|?*^]', "_", s)
    return s[:80] or "query"


# -------------------------
# Function: Query Wolfram Alpha
# -------------------------
def wolfram_query(query, include_step_by_step=True):
    params = {
        "appid": APP_ID,
        "input": query,
        "output": "JSON",
        "format": "plaintext,image"
    }

    # Add step-by-step if requested
    if include_step_by_step:
        params["podstate"] = "Step-by-step solution"

    response = requests.get(API_URL, params=params)

    if response.status_code == 401:
        raise Exception(
            "HTTP 401: Invalid or missing Wolfram Alpha App ID. "
            "Get a free App ID at https://developer.wolframalpha.com/ and set APP_ID in this file."
        )
    if response.status_code != 200:
        raise Exception(f"HTTP Error: {response.status_code}")

    return response.json()


# -------------------------
# Function: Extract and Structure JSON
# -------------------------
def extract_wolfram_steps(query, wolfram_json):
    steps_data = {"query": query, "pods": []}

    pods = wolfram_json.get("queryresult", {}).get("pods", [])
    for pod in pods:
        pod_entry = {
            "title": pod.get("title"),
            "primary": pod.get("primary", False),
            "subpods": []
        }

        for i, subpod in enumerate(pod.get("subpods", [])):
            step_entry = {}
            # Plaintext step
            if "plaintext" in subpod and subpod["plaintext"]:
                step_entry["step_index"] = i
                step_entry["plaintext"] = subpod["plaintext"]

            # Image URL (optional, useful for rendering)
            if "img" in subpod:
                step_entry["img_src"] = subpod["img"]["src"]

            if step_entry:
                pod_entry["subpods"].append(step_entry)

        steps_data["pods"].append(pod_entry)

    # Store numeric results if available
    definite_pods = [p for p in pods if "Definite integral" in p.get("title", "")]
    if definite_pods:
        steps_data["definite_result"] = definite_pods[0]["subpods"][0].get("plaintext", "")

    return steps_data


# -------------------------
# Function: Download images and add local paths
# -------------------------
def download_images(query, structured_data):
    """Download all img_src URLs and set img_local to the saved file path."""
    safe_name = _safe_folder_name(query)
    image_dir = os.path.join(OUTPUT_DIR, IMAGES_SUBDIR, safe_name)
    os.makedirs(image_dir, exist_ok=True)

    headers = {
        "User-Agent": "WolframAlpha-Python-Client/1.0 (Educational)"
    }
    idx = 0
    for pod in structured_data.get("pods", []):
        for sub in pod.get("subpods", []):
            url = sub.get("img_src")
            if not url:
                continue
            ext = ".gif"  # Wolfram typically returns GIFs
            local_name = f"{idx}{ext}"
            local_path = os.path.join(image_dir, local_name)
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 200:
                    with open(local_path, "wb") as f:
                        f.write(r.content)
                    sub["img_local"] = os.path.join(IMAGES_SUBDIR, safe_name, local_name)
                    print(f"  Saved image: {sub['img_local']}")
                else:
                    print(f"  Skip image (HTTP {r.status_code}): {url[:60]}...")
            except Exception as e:
                print(f"  Failed to download image: {e}")
            idx += 1
    return structured_data


# -------------------------
# Function: Save JSON
# -------------------------
def save_json(query, structured_data):
    safe_name = _safe_folder_name(query)
    filename = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(filename, "w") as f:
        json.dump(structured_data, f, indent=4)
    print(f"Saved structured JSON: {filename}")
    return filename


# -------------------------
# Main Function: Query + Store
# -------------------------
def store_wolfram_query(query, download_pod_images=True):
    wolfram_json = wolfram_query(query)
    structured_data = extract_wolfram_steps(query, wolfram_json)
    if download_pod_images:
        print("Downloading pod images...")
        structured_data = download_images(query, structured_data)
    save_json(query, structured_data)
    return structured_data


# -------------------------
# Example Usage
# -------------------------
if __name__ == "__main__":
    # Use query from command line, or default example
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "sqrt((34*52 + 73) - 144/4) + ln(e^5) + cos(0)^2"
    print(f"Query: {query}\n")
    data = store_wolfram_query(query)
    print("\nPrimary Pod Example Output:")
    for pod in data["pods"]:
        if pod["primary"]:
            for sub in pod["subpods"]:
                print(sub.get("plaintext", ""))
