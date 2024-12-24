import os
import ftfy
import zipfile
import json
import re
from striprtf.striprtf import rtf_to_text

output_laws = {}

def reformat_article_name(article_name):
    # Match the article format with optional hyphen
    match = re.match(r"(第\s*\d+)(?:-(\d+))?\s*條", article_name)
    if match:
        base_article = match.group(1).strip()  # "第2" in "第2-1條"
        sub_article = match.group(2)           # "1" in "第2-1條"
        # Reformat as "第2條之1" if sub_article exists
        if sub_article:
            return f"{base_article}條之{sub_article}"
        else:
            return f"{base_article}條"
    return article_name  # Return original if no match

def parse_law_content(file_name, content):
    # Split content by "第 X 條" (articles)
    articles = re.split(r"(第\s*\d+(?:-\d+)?\s*條)", content)
    
    # Iterate through the split content to structure it
    current_article = None
    law_name = None
    for part in articles:
        # Check if the part is an article identifier (e.g., "第 1 條")
        if re.match(r"第\s*\d+(?:-\d+)?\s*條", part):
            current_article = part.strip()  # Set as the current articleˊ
            law_name = reformat_article_name(current_article)
        
        elif current_article:
            content = ""
            part = part.split("\n")

            for p in part:
                content += p
            # output_laws[(file_name + law_name).replace(" ", "")] = [content]
            output_laws[(file_name + law_name).replace(" ", "")] = content
    return

def unzip_law(zip_file_path, extracted_folder):
    '''
    zip_file_path: file path to the zip file
    extracted_folder: destination folder for the extracted files

    Returs: 
        path to unzipped_directory
    '''

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)

    #iterate law_dir to fix the law's name with ftfy
    law_dir = os.path.join(extracted_folder, 'law')
    for root, dirs, files in os.walk(law_dir):
        for file in files:
            corrected_name = ftfy.fix_text(file)

            original_path = os.path.join(root, file)
            corrected_path = os.path.join(root, corrected_name)

            os.rename(original_path, corrected_path)

    return law_dir

def construct_law_data(laws_dir):
    """
    laws_dir: directory containing the law files (.zip)

    create a .jsonl file to store the structured laws_data
    """

    #iterate over each .zip file in law_dir
    for law_file_name in os.listdir(laws_dir):
        if law_file_name.endswith('.zip'):
            law_path = os.path.join(laws_dir, law_file_name)

            #open .zip file
            with zipfile.ZipFile(law_path, 'r') as law_ref:                
                with law_ref.open(law_ref.namelist()[0]) as file:
                    rtf_content = file.read().decode('utf-8')
                    #convert rtf to plain text
                    text_content = rtf_to_text(rtf_content)

                    #parse law_content
                    law_content = parse_law_content(law_file_name.strip(".zip").strip(), text_content)
    
    # write the data to a jsonl file

def add_train_data(train_data):
    with open(train_data, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            
            title = entry["title"]
            question = entry["question"]
            if question is None: question = ""
            content = title + question
            content = content.replace('"', "'")
            labels = []
            for label in entry["label"].split(','):
                label = label.strip()
                try:
                    # output_laws[label].append(content)
                    output_laws[label] = output_laws[label] + content
                except:
                    # output_laws[label] = [content]
                    output_laws[label] = content

if __name__ == '__main__':
    zip_file_path = './law.zip'
    extracted_folder = './extracted_laws'
    train_data = "./train_data.jsonl"
    
    laws_dir = unzip_law(zip_file_path, extracted_folder)
    construct_law_data(laws_dir)

    add_train_data(train_data)

    with open('laws_data.py', 'w', encoding='utf-8') as f:
        f.write("laws = [\n")
        for law_id, content in output_laws.items():
            # for content in contents:
                # content = ''.join(content.split('\n'))
                # content = ''.join(content.split('\r'))
                
                content = ''.join(content.split('\n'))
                content = ''.join(content.split('\r'))
                f.write(f'\t"{law_id}+-+-{content}",\n')
        f.write(']')