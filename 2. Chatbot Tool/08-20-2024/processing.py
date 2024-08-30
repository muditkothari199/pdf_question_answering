import pytesseract
from pdf2image import convert_from_path
import cohere
import re
import os
import tkinter as tk
from tkinter import filedialog
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation

# Set paths
pytesseract.pytesseract.tesseract_cmd = r"Tesseract-OCR\\tesseract.exe"
poppler_path = r"poppler-24.07.0\\Library\bin"

# Initialize Cohere client
cohere_api_key = '0VBE3VyqTeKVmbxEYE2t9sOEKroR4ZG11SrD6v6M'
co = cohere.Client(cohere_api_key)

updated_dict = {
    'Application': ['Application'],
    'Audit': ['Audit', 'Return Premium', 'Additional Premium'],
    'Auto ID Card': ['ID Card', 'VIN', 'Auto ID Card'],
    'Billing': ['Bill Setup', 'Policy Bill', 'Card/ACH/Method Setup'],
    'Binder': ['Binder', 'Effective Date'],
    'Broker of Record': ['AOR', 'BOR', 'Agency/Agency Change', 'Date', 'Lines of Business','Broker of Record'],
    'Cancellation': ['Cancellation Date', 'Date', 'Cancellation', 'Lines of Business'],
    'Cancellation Notice': ['Reason for Cancellation', 'Cancellation Eff Date', 'Lines of Business', 'Carrier'],
    'Certificate': ['Acord 25', 'Year', 'Policies', 'Lines of Business', 'Holder'],
    'Certificate Request': ['Holder Info', 'Certificate Request', 'Date'],
    'Claim': ['Loss Info', 'Claim', 'Details', 'Photos', 'Policy Report'],
    'Correspondence': ['Carrier Correspondence', 'Conversation', 'Notice'],
    'Declined By Company': ['Declination', 'Carrier Quote', 'Date'],
    'Driver Exclusion': ['Driver Exclusion', 'Form', 'Confirmation', 'Signed/Unsigned'],
    'Endorsement': ['Change', 'Endorsement', 'Date', 'Change Details'],
    'Endorsement Request': ['Set things to Change', 'Date', 'LOB'],
    'Evidence of Property': ['Acord 27', 'Date', 'Additional Interest'],
    'Incident Report': ['Incident', 'Line of Business', 'Loss'],
    'Invoice': ['Invoice', 'Premium', 'Date', 'Carrier'],
    # 'Loss Control': ['Loss Control', 'Recommendation', 'Picture', 'Improvement', 'Adjustor'],
    'Loss Run': ['Claims', 'Loss', 'Date', 'LOB'],
    'Lost Policy Release': ['Signed/Unsigned', 'Cancellation Date', 'Cancellation Request', 'Acord 35'],
    # 'Motor Vehicle Record': ['Voilation', 'Claims', 'Loss Date'],
    'Non-Renewal': ['Non-Renewal', 'Reason'],
    # 'Payments': ['Premium Info', 'Payment', 'Due Date', 'Carrier'],
    'Premium Finance': ['Premium Finance Agreement', 'Premium', 'Date', 'LOB'],
    'Proposal': ['Proposal', 'Carrier'],
    'Quote': ['Quote', 'Carrier', 'Premium', 'Date', 'LOB'],
    'Quote Request': ['Requested Policy Info', 'Date', 'LOB'],
    'Reinstatement': ['Reinstatement Date', 'Reinstatement', 'LOB', 'Carrier'],
    # 'Rewrite Policy': ['LOB', 'Rewrite Policy', 'Policy Confirmation'],
    'Renewal Notice': ['Renewal Notice', 'LOB', 'Date'],
    'Subjectivities': ['Every doc in reply and submission of quote and policy processing'],
    'Submission': ['Any doc that could be used to process policy'],
    'Underwriting Info': ['Old Policy Info', 'Doc']
}




key_list = [
        'Automobile Policy', 'Home', 'Auto', 'Homeowners Insurance', 'Renters Insurance', 'Flood Insurance',
        'Condo Insurance', 'Mobile Home Insurance', 'Travel Insurance', 'Disability Insurance',
        'Pet Insurance', 'Earthquake Insurance', 'Umbrella Insurance', "Workers' Compensation Insurance",
        'Cyber Liability Insurance', 'Business Insurance', 'General Liability Insurance',
        'Property Insurance', 'Business Interruption Insurance', 'Professional Liability Insurance',
        'Commercial Multiple Peril Policies', 'Term Life Insurance', 'Whole Life Insurance',
        'Product Liability Insurance', 'Builders Risk Insurance', 'Commercial Crime Insurance',
        'Environmental Liability Insurance', 'Marine Insurance', 'Universal Life Insurance',
        'Variable Life Insurance', 'Indexed Universal Life Insurance', 'Final Expense Insurance',
        'Survivorship Life Insurance (Second-to-Die Insurance)', 'Guaranteed Issue Life Insurance',
        'No-Exam Life Insurance', 'Health Maintenance Organization (HMO)',
        'Preferred Provider Organization (PPO)', 'Exclusive Provider Organization (EPO)',
        'Point of Service (POS)', 'High Deductible Health Plan (HDHP)', 'Catastrophic Health Insurance',
        'Medicare Part A (Hospital Insurance)', 'Medicare Part B (Medical Insurance)',
        'Medicare Part C (Medicare Advantage)', 'Medicare Part D (Prescription Drug Coverage)',
        'Childrens Health Insurance Program (CHIP)', 'Individual Health Insurance',
        'Group Health Insurance', 'Medicaid', 'Short-Term Health Insurance',
        'Critical Illness Insurance', 'Accident Insurance', 'Dental Insurance', 'Vision Insurance',
        'Long-Term Care Insurance', 'Health Savings Accounts (HSAs) and Flexible Spending Accounts (FSAs)',
        'Cyber Insurance', 'Environmental Liability Insurance',
        'Professional Liability Insurance (Errors and Omissions Insurance)', 'Medical Malpractice Insurance',
        'Legal Malpractice Insurance', 'Financial Services Malpractice Insurance',
        'Directors and Officers (D&O) Insurance', 'Kidnap and Ransom Insurance', 'Terrorism Insurance',
        'Event Cancellation Insurance', 'Fine Arts Insurance', 'Excess and Surplus Lines Insurance',"Auto",
        "Cyber Liability","Directors & Officers","Employment Practices Liability","Errors & Omissions","Excess Liability",
        "General Liability","Homeowners","Individual Health","Inland Marine","Professional Liability","Property","Umbrella",
        "Flood","Dwelling Fire","Mobile Home","Businessowners Policy","Boat","Property","Earthquake","Garage","Fire","Bond","Workers Compensation"]


def remove_stopwords_and_punctidation(text):
    words = word_tokenize(text)
    stop_words=set(stopwords.words("english"))
    # punctuation_set=set(punctuation)

    filtered_words=[word for word in words if word not in stop_words]
    filter_text=" ".join(filtered_words)
    
    return filter_text



def extract_text_from_images(images):
    return ''.join(pytesseract.image_to_string(image) for image in images)

def extract_text_from_pdf(pdf_path, max_pages=10):
    images = convert_from_path(pdf_path, poppler_path=poppler_path, first_page=1, last_page=max_pages)
    return extract_text_from_images(images)

def truncate_text(text, max_tokens=2500):
    words = text.split()
    truncated_text = ' '.join(words[:max_tokens])
    return truncated_text

def query_cohere_api(text, query):
    combined_message = f"Document Text: {text}\nUser Query: {query}"
    response = co.generate(
        model="command-r-plus",
        prompt=combined_message,
        max_tokens=500,
        temperature=0.0,
        stop_sequences=["\n"]
    )
    return response.generations[0].text.strip()

def format_answer(answer):
    return re.sub(r'\n+', '\n', answer).strip()

def find_dates(text):
    date_patterns = [
        r'\b\d{2}/\d{2}/\d{4}\b',   # Matches dates in MM/DD/YYYY format
        r'\b\d{2}/\d{2}/\d{2}\b',
        r'\b\d{1}/\d{2}/\d{4}\b',
        r'\b\d{4}-\d{2}-\d{2}\b',   # Matches dates in YYYY-MM-DD format
        r'\b\d{2}-\d{2}-\d{4}\b',   # Matches dates in DD-MM-YYYY format
        r'\b\d{1,2} \b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b \d{4}',  # Matches dates like 1 Jan 2023
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b \d{1,2}, \d{4}',  # Matches dates like Jan 1, 2023
        r'\b\d{1,2} \b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b \d{4}',  # Matches dates like 1 January 2023
        r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b \d{1,2}, \d{4}',  # Matches dates like January 1, 2023
        r'FROM: \d{2}-\d{2}-\d{2} TO: \d{2}-\d{2}-\d{2}',  # Matches dates like FROM: MM-DD-YY TO: MM-DD-YY
        r'\b\d{2}-\d{2}-\d{2}\b to \b\d{2}-\d{2}-\d{2}\b'  # Matches dates like MM-DD-YY to MM-DD-YY
    ]
    dates = []
    for pattern in date_patterns:
        dates.extend(re.findall(pattern, text))
    return dates

def process_dates(dates_found):
    if not dates_found:
        return 'No Date Found'

    date = dates_found[0]
    year_pattern1 = r'\d{4}'
    year_pattern2=r'\b\d{2}/\d{2}/\d{2}\b'
    match = re.search(year_pattern1, date)
    match1=re.search(year_pattern2, date)

    
    if match:
        year = match.group(0)
        last_two_digits = year[-2:]
        try:
            next_year = str(int(last_two_digits) + 1).zfill(2)
            return f"{last_two_digits}-{next_year}"
        except ValueError:
            return 'Invalid-Date-Format'
    elif match1:
         year = match1.group(0)
         last_two_digits = year[-2:]
         try:
            next_year = str(int(last_two_digits) + 1).zfill(2)
            return f"{last_two_digits}-{next_year}"
         except ValueError:
            return 'Invalid-Date-Format'
    return 'Date Format Invalid'

def find_doc_type(text):
    type_docs= query_cohere_api(truncate_text(text), " What is type of given document:? ")
    print(type_docs)
    for key,value in updated_dict.items():
        for i in value:
            if i.lower() in type_docs.lower():
               return key
    # for i in updated_dict:
    #     if i.lower() in type_docs.lower():
    #         return i


def find_premium(text):
    premium = query_cohere_api(truncate_text(text),"what is the return of the premium:?")
    print(premium)
    return premium


def get_policy_type(file_path,document_text):
    # document_text = extract_text_from_pdf(file_path)
    truncated_text = truncate_text(document_text)
    policy_type = query_cohere_api(truncated_text,  "what is the policy Type ?")

    # Check if the policy type matches any of the keys in lob_keys
    matched_policy = None
    for key in key_list:
        if key.lower() in policy_type.lower():
            matched_policy = key.strip()
            break
    
    if matched_policy:
        return matched_policy
    else:
        return policy_type

def process_pdfs_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {filename}...")

            # Extract text from PDF
            pdf_text = extract_text_from_pdf(file_path)

            # If the extracted text is empty, use OCR
            try:
                if not pdf_text.strip():
                    print(f"No text found in {filename}. Using OCR...")
                    pdf_text = extract_text_from_images(convert_from_path(file_path, poppler_path=poppler_path))
            except Exception as e:
                print(f"Error using OCR: {e}")
                continue
            # print(pdf_text)
            # print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            pdf_text=remove_stopwords_and_punctidation(pdf_text)
            # print(pdf_text)
            # Extract and process dates
            dates_found = find_dates(pdf_text)
            processed_date = process_dates(dates_found)
            if processed_date == 'No Date Found':
                str1 = "0-0"
            else:
                year = processed_date.split('-')[0]
                x = int(year) + 1
                str1 = f"{year}-{x}"
            
            # Find carrier name
            doc_type = find_doc_type(pdf_text)

            pre = find_premium(pdf_text)

            # Get policy type
            policy_type = get_policy_type(file_path,pdf_text)
            if policy_type == "Unknown":
                policy_type = "Unknown"
            print("policy type", policy_type)
            print("doc_type",doc_type)
            print("pre",pre)
            new_name = f"{str1} {policy_type}-Policy-{doc_type}.pdf"
            print(new_name)

            """
            # Generate new file name and rename
            new_name = f"{str1} {policy_type}-Policy-{carrier}.pdf"
            new_file_path = os.path.join(folder_path, new_name)

            # Check for file name collisions
            base, extension = os.path.splitext(new_file_path)
            counter = 1
            while os.path.exists(new_file_path):
                new_file_path = f"{base}_{counter}{extension}"
                counter += 1

            os.rename(file_path, new_file_path)

            # Print the old and new names
            print(f"Old Name: {filename} -> New Name: {new_name}")
          """
# Tkinter setup for folder selection
root = tk.Tk()
root.withdraw()  # Hide the root window

folder_path = filedialog.askdirectory(title="Select Folder Containing PDFs")
if folder_path:
    process_pdfs_in_folder(folder_path)
else:
    print("No folder selected.")










