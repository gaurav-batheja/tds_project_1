from fastapi import FastAPI, HTTPException, Query
import openai
import os
import glob
import json
import subprocess
import sqlite3
import re
import builtins
import shutil
import inspect
import base64
import requests
import psutil
import urllib
import shutil
from dotenv import load_dotenv
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from fastapi.responses import FileResponse
from word2number import w2n
from datetime import datetime
from dateutil import parser  # ✅ Automatically detects date formats
app = FastAPI()
load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"myenv\Lib\site-packages\Tesseract-OCR\tesseract.exe"
# Set OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("AIPROXY_TOKEN")
if not OPENAI_API_KEY:
    raise ValueError("Missing OpenAI API key. Set it in environment variables.")

openai.api_key = OPENAI_API_KEY
openai.base_url = "https://aiproxy.sanand.workers.dev/openai/v1/".rstrip("/") + "/"  # ✅ Ensure correct URL format

# Task Parsing Prompt
TASK_PROMPT = """
You are an AI agent that extracts structured information from task descriptions.
Given a task description, return JSON with:
- "task_type": (e.g., "count_days", "format_file", "sort_contacts", "extract_recent_logs")
- "input_file": Path of the input file (if applicable, relative to the current directory)
- "output_file": Path of the output file (if applicable, relative to the current directory)
- Other required parameters (e.g., "day" for counting, "sort_keys" for sorting, "ticket_type" for querying sales)
- "num_logs": (integer, optional, defaults to 10) - The number of recent log files to read
- "line_number": (string, can be "first", "second", "last") - The line to extract


### Rules:
- If the task is about **installing `uv` and running `datagen.py`**, classify it as **`run_datagen`**.
- If the task is about **database queries**, classify it as **`query_sqlite`**.
- If the task involves **formatting a file**, set `"task_type": "format_file"`.
- If the task involves **comments**, set `"task_type": "find_most_similar_comments"`.
- If the task is about extracting **H1** from Markdown files, set `"task_type": "extract_h1_titles"` and set `"docs_dir": "input_file"`.
- If the task involves **extracting from logs**, set `"task_type": "extract_recent_logs" and always set `"logs_dir": "input_file"`.
- If the task involves **counting occurrences of a day**, set `"task_type": "count_days"`.
- If the task involves **extract email address**, set `"task_type": "extract_email"`.
- If the task involves **querying a SQLite database**, set `"task_type": "query_sqlite"`.
- If the task involves **extracting the first line of recent logs**, set `"line_number": "first"`.
- If the task asks for **the second line**, set `"line_number": "second"`.
- If the task asks for **the last line**, set `"line_number": "last"`.
- If an email id is provided, set it to "email".
- If no number is provided, set `"num_logs": 10`.
- If the task says all lines do accordingly.
- If the task involves **processing or extracting details from a credit card image**, set `"task_type": "extract_credit_card"`.
 - Example: "Extract credit card details from `data/credit_card.png`"
- **Do NOT classify "extract" as "format_file"** unless the task explicitly mentions formatting.
- Ensure file paths are correctly extracted **without duplicating base directories** (e.g., `"data/data/...` → `"data/...`").
- Extract and return the structured JSON. **No explanations, no extra formatting.**
Task Description:
{task}

Extract and return the structured JSON. No explanations, no extra formatting.
"""

# 🔹 Constants for A1
DATAGEN_URL = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"
SCRIPT_PATH = "./datagen.py"  # Local script path

# 🔹 Security Constants
DATA_DIR = "./data"


def install_uv():
    """Check if `uv` is installed and install it if missing."""
    if shutil.which("uv") is None:  # Check if `uv` is in PATH
        try:
            subprocess.run(["pip", "install", "uv"], check=True, capture_output=True, text=True)
            return "uv installed successfully."
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to install uv: {e.stderr}")
    return "uv is already installed."

def download_datagen():
    """Download `datagen.py` if it doesn't exist in the same directory as `app.py`."""
    if not os.path.exists(SCRIPT_PATH):
        try:
            subprocess.run(["curl", "-o", SCRIPT_PATH, DATAGEN_URL], check=True, capture_output=True, text=True)
            subprocess.run(["chmod", "+x", SCRIPT_PATH])  # Ensure script is executable
            return "datagen.py downloaded successfully."
        except subprocess.CalledProcessError as e:
            raise HTTPException(status_code=500, detail=f"Failed to download datagen.py: {e.stderr}")
    return "datagen.py already exists."

# def download_datagen():
#     """Download `datagen.py` if it doesn't exist."""
#     if not os.path.exists(SCRIPT_PATH):
#         try:
#             subprocess.run(["curl", "-o", SCRIPT_PATH, DATAGEN_URL], check=True, capture_output=True, text=True)
#             return "datagen.py downloaded successfully."
#         except subprocess.CalledProcessError as e:
#             raise HTTPException(status_code=500, detail=f"Failed to download datagen.py: {e.stderr}")
#     return "datagen.py already exists."

def run_datagen(email: str):
    """Run `datagen.py` using `uv run` with the hardcoded email."""
    try:
        result = subprocess.run(["uv", "run", SCRIPT_PATH, email, "--root", "./data"], capture_output=True, text=True)

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Error executing datagen.py: {result.stderr}")

        return {"status": "success", "output": result.stdout}

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error running command: {e.stderr}")





def convert_word_to_number(word: str) -> int:
    """Converts number words like 'five' or 'third' into integers."""
    word = word.lower().strip()
    
    # ✅ Convert words like "five" → 5
    try:
        return w2n.word_to_num(word)
    except ValueError:
        pass  # Continue if not a number word

    # ✅ Convert "first" → 1, "second" → 2, "third" → 3, etc.
    ordinal_map = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10
    }
    
    return ordinal_map.get(word, None)



# 🔴 Block `os.remove()` and `shutil.rmtree()`
def blocked_delete(*args, **kwargs):
    raise HTTPException(status_code=403, detail="File deletion is not allowed.")

os.remove = blocked_delete
shutil.rmtree = blocked_delete
builtins.open = open  # ✅ Allow normal file operations (reading/writing)


def get_absolute_path(path: str):
    """Ensures all file operations stay inside the `/data/` directory."""
    base_dir = os.path.abspath("./data")  # ✅ Restrict to `/data/`
    
    # ✅ Ensure we don't double prefix `/data/`
    if path.startswith("/data/"):  
        path = path.replace("/data/", "", 1)

    abs_path = os.path.abspath(os.path.join(base_dir, path.lstrip("/")))
    
    if path.startswith("data/"):  
        path = path.replace("data/", "", 1)

    abs_path = os.path.abspath(os.path.join(base_dir, path.lstrip("/")))

    # 🔴 Security Check: Block access outside `/data/`
    if not abs_path.startswith(base_dir):
        raise HTTPException(status_code=403, detail=f"Access denied: {path} is outside /data/")

    return abs_path


# ✅ Define required parameters for each task
TASK_REQUIRED_PARAMS = {
    "run_datagen": ["email"],
    "format_file": ["input_file", "output_file"],
    "count_days": ["input_file", "output_file", "day"],
    "sort_contacts": ["input_file", "output_file"],
    "extract_recent_logs": ["output_file", "num_logs", "line_number"],
    "extract_email": ["input_file", "output_file"],
    "query_sqlite": ["input_file", "output_file", "ticket_type"],
    "extract_credit_card": ["input_file", "output_file"],
    "extract_h1_titles": ["output_file"],
    "find_most_similar_comments": ["input_file", "output_file"]
}

def parse_task(task_description: str):
    """Uses GPT-4o Mini to parse a task description into structured data."""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": TASK_PROMPT.format(task=task_description)}]
        )
        
        raw_output = response.choices[0].message.content.strip()
        print("\n🔹 Raw LLM Response:", raw_output)  # ✅ Debugging Output

        # ✅ Remove ```json ... ``` formatting if present
        cleaned_output = re.sub(r"```json\n(.*?)\n```", r"\1", raw_output, flags=re.DOTALL).strip()

        # ✅ Ensure response is valid JSON
        structured_data = json.loads(cleaned_output)

        # ✅ Validate required parameters
        task_type = structured_data.get("task_type", "").strip()
        if task_type not in TASK_REQUIRED_PARAMS:
            raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")

        missing_params = [p for p in TASK_REQUIRED_PARAMS[task_type] if p not in structured_data]
        if missing_params:
            raise HTTPException(status_code=400, detail=f"Missing required parameters for {task_type}: {missing_params}")

        return structured_data  

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail=f"Invalid JSON from LLM: {cleaned_output}")  
    except openai.OpenAIError as e:
        raise HTTPException(status_code=400, detail=f"LLM Parsing Error: {str(e)}")
# def parse_task(task_description: str):
#     """Uses GPT-4o Mini to parse the task description into structured data."""
#     try:
#         response = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": TASK_PROMPT.format(task=task_description)}]
#         )
        
#         raw_output = response.choices[0].message.content.strip()
#         print("\n🔹 Raw LLM Response:", raw_output)  # ✅ Debugging Output

#         # ✅ Remove ```json ... ``` formatting if present
#         cleaned_output = re.sub(r"```json\n(.*?)\n```", r"\1", raw_output, flags=re.DOTALL).strip()

#         # ✅ Ensure the response is valid JSON
#         structured_data = json.loads(cleaned_output)
#         return structured_data  

#     except json.JSONDecodeError:
#         raise HTTPException(status_code=500, detail=f"Invalid JSON from LLM: {cleaned_output}")  # ✅ Show exact issue
#     except openai.OpenAIError as e:
#         raise HTTPException(status_code=400, detail=f"LLM Parsing Error: {str(e)}")



def count_day_occurrences(input_file: str, target_day: str, output_file: str):
    """Counts occurrences of a specific day (e.g., Wednesday) in a date file."""
    input_file, output_file = get_absolute_path(input_file), get_absolute_path(output_file)

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"File {input_file} not found.")
    
    target_day = target_day.title()  # ✅ Standardize (e.g., "sunday" -> "Sunday")
    count = 0
    #valid_dates_found = False  # ✅ Track if we found valid dates

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            date_str = line.strip()
            if not date_str:
                continue  # Skip empty lines

            try:
                date_obj = parser.parse(date_str)  # ✅ Auto-detect date format
            #    valid_dates_found = True
                if date_obj.strftime("%A") == target_day:  # ✅ Check weekday
                    count += 1
            except (ValueError, OverflowError):
                continue  # ✅ Skip invalid dates without crashing

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(str(count))

    return {"status": "success", "message": f"Count of {target_day}: {count} written to {output_file}"}

# def count_day_occurrences(input_file: str, target_day: str, output_file: str):
#     """Counts occurrences of a specific day in a file."""
#     input_file = get_absolute_path(input_file)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(input_file):
#         raise HTTPException(status_code=404, detail=f"File {input_file} not found in project directory.")

#     with open(input_file, "r", encoding="utf-8") as f:
#         content = f.readlines()

#     count = sum(1 for line in content if target_day.lower() in line.lower())

#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write(str(count))

#     return {"status": "success", "message": f"Count of {target_day}: {count} written to {output_file}"}


def format_file(input_file: str):
    """Formats a file using Prettier (installed inside venv)."""
    input_file = get_absolute_path(input_file)

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"File {input_file} not found.")

    # ✅ Use `npx prettier` instead of trying to execute it directly
    try:
        # ✅ Run Prettier with explicit Markdown formatting options
        result = subprocess.run(
            # ["npx", "prettier@3.4.2", "--stdin-filepath", input_file],
            # capture_output=True,
            # text=True,
            # check=True,
            ["npx", "prettier@3.4.2", "--write", input_file, "--prose-wrap", "preserve"],
            capture_output=True, text=True, check=True,
            shell=True  # Ensure shell execution works across platforms
        )
        
        
        return {"status": "success", "message": f"Formatted {input_file}"}
    
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Prettier failed: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")



# def format_file(input_file: str):
#     """Formats a file using Prettier."""
#     input_file = get_absolute_path(input_file)

#     if not os.path.exists(input_file):
#         raise HTTPException(status_code=404, detail=f"File {input_file} not found.")

#     prettier_path = os.path.join(os.getcwd(), "node_modules", ".bin", "prettier")
#     subprocess.run([prettier_path, "--write", input_file], check=True)
#     return {"status": "success", "message": f"Formatted {input_file}"}


def sort_contacts(input_file: str, output_file: str):
    """Sorts contacts in a JSON file by last_name, then first_name."""
    input_file = get_absolute_path(input_file)
    output_file = get_absolute_path(output_file)

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"File {input_file} not found.")

    with open(input_file, "r", encoding="utf-8") as f:
        contacts = json.load(f)

    sorted_contacts = sorted(contacts, key=lambda x: (x.get("last_name", ""), x.get("first_name", "")))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_contacts, f, indent=2)

    return {"status": "success", "message": f"Sorted contacts saved to {output_file}"}



def extract_email(input_file: str, output_file: str):
    """Extracts sender's email from an email file using LLM."""
    input_file = get_absolute_path(input_file)
    output_file = get_absolute_path(output_file)

    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"File {input_file} not found.")

    with open(input_file, "r", encoding="utf-8") as f:
        email_content = f.read()

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the sender's email address from the email content."},
            {"role": "user", "content": email_content}
        ]
    )

    raw_output = response.choices[0].message.content.strip()

    # ✅ Use regex to extract the email (if LLM includes extra text)
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", raw_output)
    email_address = email_match.group(0) if email_match else raw_output  # ✅ Ensure only the email is extracted

# ✅ Write clean email to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(email_address)


    return {"status": "success", "message": f"Extracted email saved to {output_file}"}






def query_sqlite(db_path: str, output_file: str, ticket_type: str):
    """Computes total revenue for a specific ticket type, ignoring case, and writes to file."""
    db_path = get_absolute_path(db_path)
    output_file = get_absolute_path(output_file)

    if not os.path.exists(db_path):
        raise HTTPException(status_code=404, detail=f"Database {db_path} not found.")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # ✅ Ensure table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
        if not cursor.fetchone():
            raise HTTPException(status_code=500, detail="Table 'tickets' does not exist.")

        # ✅ Ensure correct columns exist
        cursor.execute("PRAGMA table_info(tickets);")
        columns = {col[1] for col in cursor.fetchall()}
        if not {"type", "units", "price"}.issubset(columns):
            raise HTTPException(status_code=500, detail="Missing required columns in 'tickets' table.")

        # ✅ Compute total revenue (units * price)
        ticket_type = ticket_type.lower()
        cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = ?", (ticket_type,))
        total_sales = cursor.fetchone()[0] or 0.0  # ✅ Handle NULL values

        conn.close()

        # ✅ Write result to output file
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(str(round(total_sales, 2)))  # ✅ Ensure proper rounding

        return {"status": "success", "message": f"Total revenue for '{ticket_type}': {total_sales} written to {output_file}"}

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"SQLite error: {str(e)}")

# def query_sqlite(db_path: str, output_file: str, ticket_type: str):
#     """Computes total sales (units * price) for a specific ticket type, ignoring case, and writes to file."""
#     db_path = get_absolute_path(db_path)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(db_path):
#         raise HTTPException(status_code=404, detail=f"Database {db_path} not found.")

#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         # ✅ Check if table exists
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
#         if not cursor.fetchone():
#             raise HTTPException(status_code=500, detail="Table 'tickets' does not exist.")

#         # ✅ Check if columns exist
#         cursor.execute("PRAGMA table_info(tickets);")
#         columns = {col[1] for col in cursor.fetchall()}
#         if not {"type", "units", "price"}.issubset(columns):
#             raise HTTPException(status_code=500, detail="Missing required columns in 'tickets' table.")

#         # ✅ Convert ticket type to lowercase in both query & database
#         ticket_type = ticket_type.lower()  # ✅ Normalize input case

#         # ✅ Compute total sales (units * price)
#         cursor.execute("SELECT SUM(units * price) FROM tickets WHERE LOWER(type) = ?", (ticket_type,))
#         total_sales = cursor.fetchone()[0] or 0  # ✅ Handle NULL case

#         conn.close()

#         # ✅ Write result to output file
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(str(round(total_sales, 2)))  # ✅ Ensure rounding to match expected output

#         return {"status": "success", "message": f"Total sales for '{ticket_type}': {total_sales} written to {output_file}"}

#     except sqlite3.Error as e:
#         raise HTTPException(status_code=500, detail=f"SQLite error: {str(e)}")






# def query_sqlite(db_path: str, output_file: str, ticket_type: str):
#     """Computes total sales for a given ticket type and writes to file."""
#     db_path = get_absolute_path(db_path)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(db_path):
#         raise HTTPException(status_code=404, detail=f"Database {db_path} not found.")

#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         # ✅ Check if table exists
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
#         if not cursor.fetchone():
#             raise HTTPException(status_code=500, detail="Table 'tickets' does not exist in database.")

#         # ✅ Run query & log output for debugging
#         cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = ?", (ticket_type,))
#         result = cursor.fetchone()
        
#         total_sales = result[0] if result and result[0] is not None else 0  # ✅ Handle NULL results
#         print(f"\n🔹 Query Result: {total_sales}")  # ✅ Debugging Output

#         conn.close()

#         # ✅ Write to output file
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(str(total_sales))

#         return {"status": "success", "message": f"Total sales for '{ticket_type}': {total_sales} written to {output_file}"}

#     except sqlite3.Error as e:
#         raise HTTPException(status_code=500, detail=f"SQLite error: {str(e)}")


# def query_sqlite(db_path: str, output_file: str, ticket_type: str):
#     """Computes total tickets sold for a given ticket type and writes to file."""
#     db_path = get_absolute_path(db_path)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(db_path):
#         raise HTTPException(status_code=404, detail=f"Database {db_path} not found.")

#     try:
#         conn = sqlite3.connect(db_path)
#         cursor = conn.cursor()

#         # ✅ Debug: Check if table exists
#         cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets';")
#         if not cursor.fetchone():
#             raise HTTPException(status_code=500, detail="Table 'tickets' does not exist in database.")

#         # ✅ Debug: Show sample data
#         cursor.execute("SELECT * FROM tickets WHERE type = ? LIMIT 5;", (ticket_type,))
#         sample_data = cursor.fetchall()
#         print(f"\n🔹 Sample Data for '{ticket_type}': {sample_data}")  # ✅ Debugging Output

#         # ✅ Fix: Query only sums `units` (not price)
#         cursor.execute("""
#             SELECT COALESCE(SUM(units), 0) 
#             FROM tickets 
#             WHERE type = ?
#         """, (ticket_type,))
#         total_tickets = cursor.fetchone()[0]  # ✅ Ensures total_tickets is never NULL

#         print(f"\n🔹 Total Tickets Sold Calculation: {total_tickets}")  # ✅ Debugging Output

#         conn.close()

#         # ✅ Write to output file
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(str(total_tickets))

#         return {"status": "success", "message": f"Total tickets sold for '{ticket_type}': {total_tickets} written to {output_file}"}

#     except sqlite3.Error as e:
#         raise HTTPException(status_code=500, detail=f"SQLite error: {str(e)}")



# def query_sqlite(db_path: str, output_file: str):
#     """Computes total sales for 'Gold' tickets."""
#     db_path = get_absolute_path(db_path)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(db_path):
#         raise HTTPException(status_code=404, detail=f"Database {db_path} not found.")

#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     cursor.execute("SELECT SUM(units * price) FROM tickets WHERE type = 'Gold'")
#     total_sales = cursor.fetchone()[0] or 0

#     conn.close()

#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write(str(total_sales))

#     return {"status": "success", "message": f"Total sales for 'Gold': {total_sales} written to {output_file}"}







# ✅ Convert words ("one", "two", etc.) to numbers
def convert_word_to_number(word: str):
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    return word_to_num.get(word.lower(), None)  # Returns None if not found

def extract_recent_logs(logs_dir: str, output_file: str, num_logs: int = 10, line_number: str = "first"):
    """Extracts specified lines from the X most recent .log files and writes them to a file."""
    logs_dir = get_absolute_path(logs_dir)
    output_file = get_absolute_path(output_file)

    if not os.path.exists(logs_dir):
        raise HTTPException(status_code=404, detail=f"Directory {logs_dir} not found.")

    log_files = glob.glob(os.path.join(logs_dir, "*.log"))

    if not log_files:
        raise HTTPException(status_code=404, detail="No .log files found in the directory.")

    # ✅ Sort log files by last modified time (newest first)
    log_files = sorted(log_files, key=os.path.getmtime, reverse=True)

    # ✅ Convert `line_number` to lowercase safely
    line_number_str = str(line_number).lower().strip()

    # ✅ Default to "first" if input is empty
    if not line_number_str:
        line_number_str = "first"

    # ✅ Handle queries like "first five lines", "last line", etc.
    words = line_number_str.split()
    
    # Extract number if present (digit or word)
    if len(words) > 1:
        if words[1].isdigit():
            line_number = int(words[1])  # "first 5 lines" → 5
        else:
            line_number = convert_word_to_number(words[1])  # "first five lines" → 5
    else:
        line_number = 1  # Default to single line if unspecified

    # ✅ Handle "first" and "last"
    if words[0] == "last":
        log_files.reverse()  # ✅ Reverse log selection for "last"
    elif words[0] == "all":
        line_number = "all"

    extracted_lines = []

    # ✅ Read the specified lines of up to `num_logs` logs
    for log_file in log_files[:num_logs]:  
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                # lines = f.readlines()
                lines = [line.strip() for line in f.readlines() if line.strip()]
                
                if line_number == "all":
                    selected_line = "\n".join(lines).strip()  # ✅ Return entire file
                elif isinstance(line_number, int):
                    selected_line = "\n".join(lines[:line_number]).strip()  # ✅ Get first `N` lines
                elif line_number_str == "last":
                    selected_line = lines[-1].strip() if lines else "EMPTY FILE"
                else:
                    selected_line = "UNKNOWN LINE REQUEST"

                extracted_lines.append(selected_line)   # ✅ Don't include filename
        except Exception as e:
            extracted_lines.append(f"Error reading {os.path.basename(log_file)}: {str(e)}")

    # log_files = sorted(log_files, key=os.path.basename, reverse=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # print("=== DEBUG OUTPUT ===")
        # for i, line in enumerate(extracted_lines):
        #     print(f"{i+1}: {repr(line)}")
        # print("=== END DEBUG ===")
        f.write("\n".join(extracted_lines)) 
        # f.writelines(line + "\n" for line in extracted_lines)
        # f.write("\n".join(extracted_lines) + "\n")  # ✅ Ensures expected newlines

    return {"status": "success", "message": f"Extracted {line_number} line(s) from {num_logs} most recent logs and saved to {output_file}"}



# #THIS ONE WORKS
# def extract_recent_logs(logs_dir: str, output_file: str, num_logs: int = 10, line_number: str = "first"):
#     """Extracts the first line of the 10 most recent .log files and writes them to a file."""
#     logs_dir = get_absolute_path(logs_dir)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(logs_dir):
#         raise HTTPException(status_code=404, detail=f"Directory {logs_dir} not found.")

#     # ✅ Find all `.log` files in the directory
#     log_files = glob.glob(os.path.join(logs_dir, "*.log"))

#     if not log_files:
#         raise HTTPException(status_code=404, detail="No .log files found in the directory.")

#     # ✅ Sort log files by last modified time (newest first)
#     log_files = sorted(log_files, key=os.path.getmtime, reverse=True)

#     extracted_lines = []

#     # ✅ Read the first line of up to 10 recent logs
#     for log_file in log_files[:num_logs]:  
#         try:
#             with open(log_file, "r", encoding="utf-8") as f:
#                 line = f.readline().strip()  # ✅ Get first line
#                 extracted_lines.append(line)  # ✅ Include filename
#         except Exception as e:
#             extracted_lines.append(f"Error reading {os.path.basename(log_file)}: {str(e)}")

#     # ✅ Write results to output file
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("\n".join(extracted_lines))

#     return {"status": "success", "message": f"Extracted first lines from 10 most recent logs and saved to {output_file}"}


# def extract_recent_logs(logs_dir: str, output_file: str):
#     """Extracts the first line of the 10 most recent .log files and writes them to a file."""
#     logs_dir = get_absolute_path(logs_dir)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(logs_dir):
#         raise HTTPException(status_code=404, detail=f"Directory {logs_dir} not found.")

#     # ✅ Find all `.log` files in the directory
#     log_files = glob.glob(os.path.join(logs_dir, "*.log"))

#     if not log_files:
#         raise HTTPException(status_code=404, detail="No .log files found in the directory.")

#     # ✅ Sort log files by last modified time (newest first)
#     log_files.sort(key=os.path.getmtime, reverse=True)

#     extracted_lines = []

#     # ✅ Read the first line of up to 10 recent logs
#     for log_file in log_files[:10]:  
#         try:
#             with open(log_file, "r", encoding="utf-8") as f:
#                 first_line = f.readline().strip()  # ✅ Get first line
#                 extracted_lines.append(first_line)
#         except Exception as e:
#             extracted_lines.append(f"Error reading {log_file}: {str(e)}")

#     # ✅ Write results to output file
#     with open(output_file, "w", encoding="utf-8") as f:
#         f.write("\n".join(extracted_lines))

#     return {"status": "success", "message": f"Extracted first lines from 10 most recent logs and saved to {output_file}"}



# def extract_h1_titles(docs_dir: str, output_file: str):
#     """Extracts H1 titles from Markdown files and saves as a sorted JSON index."""
#     docs_dir = get_absolute_path(docs_dir)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(docs_dir):
#         raise HTTPException(status_code=404, detail=f"Directory {docs_dir} not found.")

#     md_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)  # ✅ Include subdirectories

#     if not md_files:
#         raise HTTPException(status_code=404, detail="No Markdown files found in the directory.")

#     index = {}

#     for md_file in sorted(md_files):  # ✅ Ensure files are processed in sorted order
#         try:
#             with open(md_file, "r", encoding="utf-8") as f:
#                 for line in f:
#                     line = line.strip()
#                     if line.startswith("# "):  # ✅ Extract first H1
#                         rel_path = os.path.relpath(md_file, docs_dir).replace("\\", "/")  # ✅ Normalize path
#                         index[rel_path] = line[2:].strip()
#                         break  # ✅ Stop after first H1

#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error reading {md_file}: {str(e)}")

#     # ✅ Sort dictionary by keys (filenames)
#     sorted_index = dict(sorted(index.items()))

#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(sorted_index, f, indent=4, ensure_ascii=False, sort_keys=True)  # ✅ Write sorted JSON

#     return {"status": "success", "message": f"Extracted H1 titles from Markdown files and saved to {output_file}"}







def extract_h1_titles(docs_dir: str, output_file: str):
    """Extracts H1 titles from Markdown files and saves as a JSON index."""
    docs_dir = get_absolute_path(docs_dir)
    output_file = get_absolute_path(output_file)

    if not os.path.exists(docs_dir):
        raise HTTPException(status_code=404, detail=f"Directory {docs_dir} not found.")

    md_files = glob.glob(os.path.join(docs_dir, "**/*.md"), recursive=True)  # ✅ Include subdirectories

    if not md_files:
        raise HTTPException(status_code=404, detail="No Markdown files found in the directory.")

    index = {}

    for md_file in sorted(md_files):  # ✅ Ensure files are processed in sorted order
        try:
            with open(md_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("# "):  # ✅ Extract first H1
                        rel_path = os.path.relpath(md_file, docs_dir).replace("\\", "/").lstrip("./")  # ✅ Normalize path
                        title = " ".join(line[2:].split()).strip()  # ✅ Normalize spaces
                        index[rel_path] = title
                        break  # ✅ Stop after first H1

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error reading {md_file}: {str(e)}")

    # ✅ Sort dictionary by keys (filenames)
    sorted_index = dict(sorted(index.items()))

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sorted_index, f, ensure_ascii=False, sort_keys=True)  # ✅ Remove indentation for exact match

    return {"status": "success", "message": f"Extracted H1 titles from Markdown files and saved to {output_file}"}







def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Image file not found: {image_path}")

# ✅ A8: Extract credit card number from an image
def extract_credit_card(input_file: str, output_file: str):
    """Extracts a credit card number from an image using GPT-4o and saves it."""

    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    # ✅ Ensure image file exists
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"Input file not found: {input_file}")

    encoded_image = encode_image(input_file)

    # ✅ GPT-4o Vision API Request
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract the 16-digit number eg. 1234 4567 8910 number from this image without spaces or any extra characters."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                    }
                ]
            }
        ]
    )

    # ✅ Extract response and clean output
    card_number = response.choices[0].message.content.strip()

    # ✅ Validate the extracted card number (Must be exactly 16 digits)
    # if not card_number.isdigit() or len(card_number) != 16:
    #     raise HTTPException(status_code=400, detail=f"Invalid extracted card number: {card_number}")

    # ✅ Write the card number to output file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(card_number)

    return {"status": "success", "message": f"Extracted credit card number saved to {output_file}"}


# def extract_credit_card(input_file: str, output_file: str):
#     """Extracts a credit card number from an image and writes it to a file."""
#     input_file = get_absolute_path(input_file)
#     output_file = get_absolute_path(output_file)

#     if not os.path.exists(input_file):
#         raise HTTPException(status_code=404, detail=f"File {input_file} not found.")

#     try:
#         # ✅ Load image
#         image = Image.open(input_file)

#         # ✅ Extract text using Tesseract OCR
#         extracted_text = pytesseract.image_to_string(image)

#         # ✅ Use regex to extract a credit card number (16 digits, ignoring spaces/hyphens)
#         match = re.search(r"\b(\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4})\b", extracted_text)
        
#         if not match:
#             raise HTTPException(status_code=400, detail="No valid credit card number found.")

#         # ✅ Clean the number (remove spaces & hyphens)
#         credit_card_number = re.sub(r"[-\s]", "", match.group(1))

#         # ✅ Write to output file
#         with open(output_file, "w", encoding="utf-8") as f:
#             f.write(credit_card_number)

#         return {"status": "success", "message": f"Extracted credit card number written to {output_file}"}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error extracting credit card: {str(e)}")



# def extract_credit_card_number(input_file: str, output_file: str):
#     """Extracts credit card number from an image and writes it to a text file."""
#     # Ensure the input path is within the allowed directory
#     input_file = get_absolute_path(input_file)
#     output_file = get_absolute_path(output_file)

#     # Check if the input image exists
#     if not os.path.exists(input_file):
#         raise HTTPException(status_code=404, detail=f"File {input_file} not found.")

#     try:
#         # Open the image using PIL
#         image = Image.open(input_file)

#         # Use Tesseract to do OCR on the image
#         extracted_text = pytesseract.image_to_string(image)

#         # Define a regex pattern for credit card numbers (simple version)
#         cc_pattern = re.compile(r'\b(?:\d[ -]*?){13,16}\b')

#         # Search for the pattern in the extracted text
#         match = cc_pattern.search(extracted_text)

#         if match:
#             # Clean the extracted number (remove spaces and hyphens)
#             credit_card_number = re.sub(r'[^\d]', '', match.group())

#             # Write the credit card number to the output file
#             with open(output_file, 'w') as f:
#                 f.write(credit_card_number)

#             return {"status": "success", "message": f"Credit card number extracted and saved to {output_file}"}
#         else:
#             return {"status": "failure", "message": "No credit card number found in the image."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")



import openai
import numpy as np
import os
from scipy.spatial.distance import cosine
from fastapi import HTTPException

def find_most_similar_comments(input_file: str, output_file: str):
    """Finds the most similar pair of comments using embeddings and writes them to output_file."""
    
    input_file = os.path.abspath(input_file)
    output_file = os.path.abspath(output_file)

    # ✅ Ensure the input file exists
    if not os.path.exists(input_file):
        raise HTTPException(status_code=404, detail=f"Input file not found: {input_file}")

    # ✅ Read comments
    with open(input_file, "r", encoding="utf-8") as f:
        comments = [line.strip() for line in f.readlines() if line.strip()]

    if len(comments) < 2:
        raise HTTPException(status_code=400, detail="Not enough comments to find a similar pair.")

    # ✅ Get embeddings using OpenAI's text-embedding-3-small
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=comments
    )
    
    embeddings = np.array([embedding.embedding for embedding in response.data])

    # ✅ Compute cosine similarity
    num_comments = len(comments)
    similarity_matrix = np.zeros((num_comments, num_comments))

    max_sim = -1
    best_pair = (None, None)

    for i in range(num_comments):
        for j in range(i + 1, num_comments):
            sim = 1 - cosine(embeddings[i], embeddings[j])
            similarity_matrix[i][j] = sim
            
            # ✅ Find most similar pair
            if sim > max_sim:
                max_sim = sim
                best_pair = (comments[i], comments[j])
                best_indices = (i, j)

    first_index, second_index = best_indices
    reversed_pair = (comments[second_index], comments[first_index])

    with open(output_file, "w", encoding="utf-8") as f:
        # sorted_pair = sorted([comments[first_index], comments[second_index]])
        f.write("\n".join(reversed_pair))
        # f.write("".join(reversed_pair).strip())
        # f.writelines(line + "\n" for line in reversed_pair)

    # # ✅ Write the most similar pair to the output file
    # with open(output_file, "w", encoding="utf-8") as f:
    #     f.write(best_pair[0] + "\n" + best_pair[1])

    return {
        "status": "success",
        "message": f"Most similar pair written to {output_file}",
        "similarity_score": max_sim,
        "selected_pair": best_pair
    }









@app.get("/read")
def read_file(path: str):
    """Reads and returns the content of a file."""
    abs_path = get_absolute_path(path)  # ✅ Ensure security

    if not os.path.exists(abs_path):
        raise HTTPException(status_code=404, detail=f"File {path} not found.")

    return FileResponse(abs_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)



@app.post("/run")
def run_task(task: str = Query(..., description="Plain-English task description")):
    """Parses the task description, executes the task, and writes the output."""
    structured_data = parse_task(task)

    task_type = structured_data.get("task_type")
    input_file = structured_data.get("input_file")  
    output_file = structured_data.get("output_file")  
    logs_dir = structured_data.get("logs_dir")  
    ticket_type = structured_data.get("ticket_type")  
    docs_dir = structured_data.get("docs_dir")  
    target_day = structured_data.get("day")  
    email = structured_data.get("email")
    
    # ✅ Extract `db_path` for A10 if missing
    if task_type == "query_sqlite" and not structured_data.get("db_path"):
        structured_data["db_path"] = structured_data.get("input_file")
        input_file = None  # ✅ Remove incorrect input_file

    # ✅ Convert `num_logs` only for log extraction tasks
    num_logs = None  
    if task_type == "extract_recent_logs":
        num_logs_raw = structured_data.get("num_logs", "10")  
        num_logs = convert_word_to_number(num_logs_raw) if isinstance(num_logs_raw, str) else num_logs_raw
    
    # ✅ Convert `line_number` only for `extract_recent_logs`
    line_number = None
    if task_type == "extract_recent_logs":
        line_number_raw = structured_data.get("line_number", "first")  
        line_number = convert_word_to_number(line_number_raw) if isinstance(line_number_raw, str) else line_number_raw


    # # ✅ Fix: Ensure email is extracted from `args` if missing
    # if task_type == "run_datagen":
    #     if not email:
    #         # ✅ Check if `args` contains email
    #         email_args = structured_data.get("arguments", [])
    #         if email_args and isinstance(email_args, list):
    #             email = email_args[0]  # ✅ Extract first argument as email

    # # ✅ Extract email if inside `"arguments"`
    # if not email and "arguments" in structured_data:
    #     email = structured_data["arguments"].get("email")

    # # ✅ Assign default email for `run_datagen`
    # if not email and task_type == "run_datagen":
    #     email = "21f1002626@ds.study.iitm.ac.in"

    # ✅ Run `datagen.py` separately for A1
    # if task_type == "run_datagen":
    #     if not email:
    #         raise HTTPException(status_code=400, detail="Missing required parameter: 'email' for run_datagen.")
    #     install_uv()
    #     download_datagen()
    #     return run_datagen(email)

    # ✅ Fix incorrect `docs_dir` assignment
    if task_type == "extract_h1_titles" and not docs_dir:
        docs_dir = input_file 

    if task_type == "extract_recent_logs" and not logs_dir:
        logs_dir = input_file 

    # if task_type == "extract_recent_logs":
    #     if not logs_dir:
    #         logs_dir = structured_data.get("input_file") or structured_data.get("logs_dir")  # ✅ Force use input_file if needed
    #         structured_data["logs_dir"] = logs_dir  # ✅ Ensure logs_dir is explicitly set
    #         structured_data["input_file"] = None  # ✅ Clear incorrect input_file

    # if not logs_dir:
    #     raise HTTPException(status_code=400, detail="Missing required parameter: 'logs_dir' for extract_recent_logs.")

    # print(f"✅ Logs Directory Set: {logs_dir}")  # ✅ Debugging output
    # # ✅ Ensure `logs_dir` is set correctly for log extraction
    # # ✅ Fix logs_dir issue
    # if task_type == "extract_recent_logs" and not logs_dir:
    #     logs_dir = input_file
    #     structured_data["logs_dir"] = logs_dir  # ✅ Ensure it's correctly assigned
    #     structured_data["input_file"] = None  # ✅ Prevent incorrect parameter usage

 
    # if task_type == "extract_recent_logs":
    #     if not logs_dir:
    #         raise HTTPException(status_code=400, detail="Missing required parameter: 'logs_dir'")
    #     print(f"🟢 DEBUG: logs_dir = {logs_dir}, output_file = {output_file}, num_logs = {num_logs}, line_number = {line_number}")


    # ✅ Task-to-Function Mapping
    task_mapping = {
        "count_days": count_day_occurrences,  
        "format_file": format_file,  
        "sort_contacts": sort_contacts,  
        "extract_email": extract_email,  
        "query_sqlite": query_sqlite,  
        "extract_recent_logs": extract_recent_logs,
        "extract_credit_card": extract_credit_card,
        "extract_h1_titles": extract_h1_titles,
        "find_most_similar_comments": find_most_similar_comments,
        "run_datagen": run_datagen
    }

    if task_type not in task_mapping:
        raise HTTPException(status_code=400, detail=f"Unknown task type: {task_type}")

    func = task_mapping[task_type]
    sig = inspect.signature(func)  

    # ✅ Extract only the relevant arguments
    params = {
        "input_file": input_file,
        "output_file": output_file,
        "logs_dir": logs_dir,  
        "target_day": target_day,
        "ticket_type": ticket_type,
        "num_logs": num_logs,  # ✅ Only for `extract_recent_logs`
        "line_number": line_number,  # ✅ Only for `extract_recent_logs`
        "docs_dir": docs_dir,
        "db_path": structured_data.get("db_path"),  # ✅ Only for `query_sqlite`
        "email": email
    }
    
    # ✅ Filter out irrelevant arguments
    valid_args = {k: v for k, v in params.items() if k in sig.parameters and v is not None}

    try:
        return func(**valid_args)  
    except TypeError as e:
        raise HTTPException(status_code=400, detail=f"Incorrect arguments for {task_type}: {str(e)}")

