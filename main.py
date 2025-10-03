import argparse
import datetime
import json
import time
import requests
import random

from typing import List, Any, Union
from groq import Groq
import form

from dotenv import load_dotenv
load_dotenv()

client = Groq()

global personality, memory
personality = {"name": "none", "email_address": "na", "personality": "dne"}
memory = []

PERSONALITY_MODEL = "openai/gpt-oss-20b"
RESPONSE_MODEL = "moonshotai/kimi-k2-instruct-0905"

def get_personality():
    prompt = """
Craft a unique personality about a person in college in India. Give them preferences about religion, following parent's advice, their behaviour, their characteristics, and give them some insecurities. Also give some viewpoints they might have on certain worldwide topics, like taste in music, and geopolitics and give them academic interests (if they exist) and creative characteristics (if they exist). Give me a detailed response of their personality, and only their personality.

It is not necessary that this person be a good person, they can be a rebellious person as well.

Return your response as a JSON object:
{
    "name": "Name of the person (only the name, can also only be the first name)",
    "email_address": "Email address of the person, with a gmail account",
    "personality": "A full detailing of the person's personality, as a string"
}
    """
    
    return client.chat.completions.create(
        model=PERSONALITY_MODEL,
        messages=[ {"role": "user", "content": prompt} ],
        temperature=2,
        max_completion_tokens=8192,
        top_p=1,
        reasoning_effort="medium",
        stream=False,
        response_format={"type": "json_object"},
    ).choices[0].message
    
def get_response(question, choices: Union[List, str], required: bool):
    sys_prompt = f""" 
You are acting on behalf of a person as an intelligent agent to fill a google form. 
Your name is {personality['name']} with the email ID {personality['email_address']}. 
Your personality is this: \n {personality['personality']}.\n\n
Act as this person and answer the questions posed to you.

CRITICAL RULES:
1. If you are given choices, your response MUST be EXACTLY one of those choices - word for word, letter for letter
2. DO NOT add explanations, context, or any other text
3. DO NOT modify the choice text in any way
4. Return ONLY the JSON: {{"response": "<exact choice>"}}

For multiple choice: Copy the exact text of one choice.
For open questions: Provide a relevant answer based on the personality.
"""
    for msg in memory:
        if msg['role'] == "user":
            sys_prompt += str(msg['content']) + " - "
        elif msg['role'] == 'assistant':
            sys_prompt += str(msg['content']) + "\n"

    usr_prompt = f"Question: {question}\n\n"
    if isinstance(choices, list):
        usr_prompt += f"AVAILABLE CHOICES (pick EXACTLY one):\n"
        for i, choice in enumerate(choices, 1):
            usr_prompt += f"{i}. {choice}\n"
        usr_prompt += f"\nYour response MUST be one of these exact texts. Copy it exactly."
    else:
        usr_prompt += f"Open question (type: {choices})"
    if required:
        usr_prompt += "\n[REQUIRED - you must answer]"

    retries = 0
    response = None

    while retries < 5:
        try:
            raw = client.chat.completions.create(
                model=RESPONSE_MODEL,
                messages=[ 
                    {"role": "system", "content": sys_prompt}, 
                    {"role": "user", "content": usr_prompt} 
                ],
                temperature=0.7,
                max_completion_tokens=512,
                top_p=0.9,
                stream=False,
                response_format={"type": "json_object"},
            ).choices[0].message.content

            try:
                parsed = json.loads(raw)
                response = parsed.get("response", "").strip()
            except Exception as e:
                print(f"[Warning] JSON parse error: {e}, raw: {raw}")
                response = None
                retries += 1
                continue

            # Validate choice-based responses
            if isinstance(choices, list) and choices:
                # Exact match
                if response in choices:
                    break
                
                # Case-insensitive match
                response_lower = response.lower()
                matched = False
                for choice in choices:
                    if str(choice).lower() == response_lower:
                        response = str(choice)
                        matched = True
                        break
                
                if matched:
                    break
                
                # Strip and retry
                for choice in choices:
                    if str(choice).strip().lower() == response_lower:
                        response = str(choice)
                        matched = True
                        break
                
                if matched:
                    break
                    
                print(f"[Retry {retries+1}/5] Got '{response}' but need one of: {choices}")
                retries += 1
                continue
            else:
                # Open-ended question
                if response:
                    break
                else:
                    print(f"[Retry {retries+1}/5] Empty response")
                    retries += 1
                    continue
                    
        except Exception as e:
            print(f"[Error] API call failed: {e}")
            retries += 1
            time.sleep(1)
            continue

    # Hard fallback - NEVER return NA for choice-based questions
    if isinstance(choices, list) and choices:
        if not response or response not in choices:
            response = random.choice(choices)
            print(f"[FORCED FALLBACK] Selected random choice: {response}")
    elif not response:
        response = "NA"
        print(f"[FORCED FALLBACK] Using NA")

    memory.extend([
        {"role": "user", "content": usr_prompt},
        {"role": "assistant", "content": str(response)}
    ])
    return response

def fill_agentic_answer(type_id, entry_id, options, required=False, entry_name=''):
    response = None
    
    if "email address" in entry_name.lower():
        return personality['email_address'] if (required or random.randint(0,100) > 50) else ""

    print(f"\n[DEBUG] type_id={type_id}, options={options}, required={required}")

    # Generate response based on type
    if type_id == 0:  # Short answer
        response = get_response(entry_name, choices='sentence', required=required)
    elif type_id == 1:  # Paragraph
        response = get_response(entry_name, choices='paragraph', required=required)
    elif type_id == 2:  # Multiple choice
        # Convert options to strings and pass them
        str_options = [str(opt) for opt in options] if options else []
        print(f"[DEBUG] MCQ - str_options={str_options}")
        if not str_options:
            print(f"[ERROR] Empty options for MCQ question: {entry_name}")
            response = "NA"
        else:
            response = get_response(entry_name, choices=str_options, required=required)
    elif type_id == 4:  # Dropdown/Checkboxes - TREAT SAME AS MULTIPLE CHOICE
        str_options = [str(opt) for opt in options] if options else []
        print(f"[DEBUG] Type 4 (dropdown/checkbox) - str_options={str_options}")
        if not str_options:
            print(f"[ERROR] Empty options for type 4 question: {entry_name}")
            response = "NA"
        else:
            response = get_response(entry_name, choices=str_options, required=required)
    elif type_id == 5:  # Linear scale
        # Convert options to strings and pass them
        str_options = [str(opt) for opt in options] if options else []
        print(f"[DEBUG] Linear scale - str_options={str_options}")
        if not str_options:
            print(f"[ERROR] Empty options for scale question: {entry_name}")
            response = "NA"
        else:
            response = get_response(entry_name, choices=str_options, required=required)

    # CRITICAL: Post-validation for choice-based fields - NEVER allow NA
    if type_id in [2, 4, 5]:  # Added type 4
        print(f"[DEBUG] Post-validation check: type_id={type_id}, options={options}")
        if options and len(options) > 0:
            valid_options_str = [str(o) for o in options]
            resp_str = str(response) if response else ""
            
            print(f"[DEBUG] Checking '{resp_str}' against {valid_options_str}")
            
            # If response is not in valid options OR is "NA", force a random choice
            if resp_str not in valid_options_str or resp_str == "NA" or not resp_str:
                print(f"[POST-VALIDATION FIX] Response '{resp_str}' invalid for '{entry_name}'")
                print(f"  Valid options: {valid_options_str}")
                response = random.choice(valid_options_str)
                print(f"  Forced to: {response}")
        else:
            print(f"[WARNING] No options provided for choice question: {entry_name}")

    # Sanitize text responses
    if response and isinstance(response, str):
        response = (response.replace('\x92', "'")
                            .replace('‟', "'")
                            .replace("'", "'")
                            .replace('"', '"')
                            .replace('"', '"')
                            .replace('—', '-')
                            .replace('\t', ' ')
                            .replace('\n', ' ')
                            .replace('\r', ' ')
                            .replace('‖', ''))
        
        # Special handling for age
        if "age" in entry_name.lower() and required and type_id == 0:
            digits = "".join([c for c in response if c.isdigit()])
            if digits:
                response = digits
            else:
                response = '25'

    # Date/time handling
    if type_id == 9:
        response = datetime.date.today().strftime('%Y-%m-%d')
    elif type_id == 10:
        response = datetime.datetime.now().strftime('%H:%M')

    # Final fallback for empty responses
    if not response:
        if type_id in [2, 4, 5] and options:  # Added type 4
            response = random.choice([str(o) for o in options])
            print(f"[EMPTY FALLBACK] Forced random choice: {response}")
        else:
            response = 'NA'

    print(f"Question: {entry_name}")
    print(f"Required: {required}")
    print(f"Choices: {options}")
    print(f"Final Response: {response}\n")
    
    return response

def generate_request_body(url: str, only_required = False):
    data = form.get_form_submit_request(
        url,
        only_required = only_required,
        fill_algorithm = fill_agentic_answer,
        output = "return",
        with_comment = False
    )
    data = json.loads(data)
    return data

def submit(url: str, data: Any):
    url = form.get_form_response_url(url)
    print("\n" + "="*60)
    print("SUBMITTING FORM")
    print("="*60)
    print(f"URL: {url}")
    print(f"Data: {json.dumps(data, indent=2)}\n")
   
    res = requests.post(url, data=data, timeout=5)
    if res.status_code != 200:
        print(f"❌ ERROR! Form submission failed with status {res.status_code}")
        print(f"Response: {res.text[:500]}")
    else:
        print("✅ Form submitted successfully!")

def main(url, only_required = False):
    try:
        payload = generate_request_body(url, only_required = only_required)
        submit(url, payload)
        print("\n✅ Done!!!\n")
    except Exception as e:
        print(f"\n❌ Error! {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Submit google form with custom data')
    parser.add_argument('url', help='Google Form URL')
    parser.add_argument("-r", "--required", action='store_true', help='If you only want to submit the required questions')
    parser.add_argument("-n", "--num", type=int, nargs=1, help="Number of responses you want", default=50)
    args = parser.parse_args()
    
    for i in range(args.num[0]):
        print(f"\n\n{'='*80}")
        print(f"SUBMISSION {i}")
        print(f"{'='*80}\n")

        retries = 0
        while True:
            try:
                personality = json.loads(get_personality().content)
                memory = []
                break
            except Exception as e:
                print(f"Personality creation failed: {e}, retrying...")
                retries += 1
                if retries == 3: 
                    raise Exception("Max retries for fetching personality exceeded")
                time.sleep(2)
                continue
                    
        print(f"Name: {personality['name']}")
        print(f"Email: {personality['email_address']}")
        print(f"Personality: {personality['personality'][:200]}...\n")
        
        main(args.url, args.required)
        
        if i < args.num[0] - 1:  # Don't sleep after last submission
            print(f"\n⏳ Sleeping for 10 seconds to avoid rate limiting...\n")
            time.sleep(10)
