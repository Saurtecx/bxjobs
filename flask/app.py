from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from flask_pymongo import PyMongo
import pymongo
import PyPDF2
import docx
import os
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from joblib import load
from nltk.tokenize import word_tokenize
import string
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk



app = Flask(__name__)
CORS(app)
app.config["MONGO_URI"] = "mongodb+srv://saurtecx:saurabhmishra@cluster0.6eo5i.mongodb.net/LinkedIn_DB?retryWrites=true&w=majority"
if not os.path.exists('./uploads'):
    os.makedirs('./uploads')
app.config['UPLOAD_FOLDER'] = './uploads'
mongo = PyMongo(app)
skill_keywords = ['Python', 'Java', 'Machine Learning', 'Data Analysis', 'C++', 'MySQL', 'Algorithms',
                   'JavaScript', 'PHP', 'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 
                   'Express.js', 'MongoDB', 'SQL', 'PostgreSQL', 'Git', 'GitHub', 'Docker', 
                   'Kubernetes', 'AWS', 'Azure', 'Google Cloud Platform', 'Linux', 'Windows', 
                   'MacOS', 'Android', 'iOS', 'Flutter', 'Django', 'Flask', 'PyTorch', 'TensorFlow', 
                   'Keras', 'Numpy', 'Pandas', 'Matplotlib', 'Seaborn', 'Scikit-learn', 'NLTK', 
                   'OpenCV', 'Unity', 'Unreal Engine', 'Blender', 'Maya', 'Photoshop', 'Illustrator', 
                   'InDesign', 'Premiere Pro', 'After Effects', 'Final Cut Pro', 'Logic Pro', 'Pro Tools', 
                   'Ableton Live', 'AutoCAD', 'Revit', 'SketchUp', 'SolidWorks', 'R', 'Scala', 'Hadoop', 
                   'Spark', 'Hive', 'Pig', 'MapReduce', 'Kafka', 'Zookeeper', 'Flume', 'Sqoop', 'Impala', 
                   'Oozie', 'Nifi', 'Tableau', 'Power BI', 'Excel', 'Google Sheets', 'Project Management', 
                   'Agile', 'Scrum', 'Waterfall', 'Jira', 'Confluence', 'Trello', 'Asana', 'Slack', 
                   'Microsoft Office', 'Google Workspace', 'Salesforce', 'Zendesk', 'HubSpot', 
                   'Mailchimp', 'AdWords', 'Analytics', 'SEO', 'SEM', 'Content Marketing', 
                   'Social Media Marketing', 'Email Marketing', 'Copywriting', 'Editing', 
                   'Proofreading', 'Technical Writing', 'Creative Writing', 'UX Design', 'UI Design', 
                   'Graphic Design', 'Web Design', 'Mobile Design', 'Product Design', 
                   'Product Management', 'Customer Service', 'Leadership', 'Teamwork', 
                   'Communication', 'Problem Solving', 'Critical Thinking', 'Time Management', 
                   'Organizational Skills', 'Multitasking', 'Negotiation', 'Sales', 'Marketing', 
                   'Business Development', 'Finance', 'Accounting', 'Human Resources', 
                   'Recruiting', 'Training', 'Coaching', 'Mentoring', 'Public Speaking', 
                   'Event Planning', 'Fundraising', 'Nonprofit Management', 'Grant Writing', 
                   'Volunteer Management', 'Project Coordination', 'Logistics', 'Supply Chain Management', 
                   'Quality Assurance', 'Quality Control', 'Risk Management', 'Compliance', 
                   'Legal', 'Contract Management', 'Intellectual Property', 'Patents', 
                   'Trademarks', 'Privacy', 'Security', 'Cybersecurity', 'Ethical Hacking', 
                   'Penetration Testing', 'IT Support', 'Network Administration', 'System Administration', 
                   'Database Administration', 'Software Testing', 'QA', 'DevOps', 
                   'Site Reliability Engineering', 'Cloud Architecture', 'Virtualization', 'Automation', 
                   'Scripting', 'Bash', 'Python Scripting', 'PowerShell', 'Ruby', 'Perl', 'C#', 'Swift', 
                   'Objective-C', 'Go', 'Rust', 'Scala']

df = pd.read_csv("jobs.csv")

def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    
    # Join filtered tokens back into a string
    text = " ".join(filtered_text)
    
    return text

df["key_skills"] = df["key_skills"].apply(preprocess_text)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["key_skills"])
# Or use TfidfVectorizer for TF-IDF feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["key_skills"])

def extract_resume_text(filename):
    # filename = file.filename
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == '.pdf':
        with open(filename, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            resume_text = ''
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                resume_text += page.extract_text()
    elif ext == '.docx':
        doc = docx.Document(filename)
        resume_text = ''
        for para in doc.paragraphs:
            resume_text += para.text
    elif ext == '.txt':
        with open(filename, 'r') as file:
            resume_text = file.read()
    else:
        raise ValueError(f'Unsupported file format: {ext}')
    
    # Preprocess the resume text
    sentences = sent_tokenize(resume_text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    skills = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        for keyword in skill_keywords:
            if keyword.lower() in stemmed_words:
                skills.append(keyword)
    
    return skills



def process_file(file):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    skills = extract_resume_text(file_path)
    os.remove(file_path)
    # skills = extract_resume_text(file)
    print('My Skills:', skills)
    # Load the model from the file
    svm_model = load('svm_model.joblib')
    key_skills = [preprocess_text(text) for text in skills]

    # Convert the text into numerical features
    # vectorizer = TfidfVectorizer()
    # vectorizer.fit(key_skills)
    X_new = vectorizer.transform(key_skills)

    # Use the trained model to predict the job title
    job_title = svm_model.predict(X_new)

    print(job_title)
    job_title_list = list(set(job_title.tolist())) 
    jobs = mongo.db.JOBS_Basic.find({
    '$or': [
        {'TITLE': {'$in': job_title_list}},  
        {'TITLE': {'$regex': f'^{job_title_list[0]}\s', '$options': 'i'}}
    ]
    })
    result = []
    print(type(jobs))
    for job in jobs:
        print(type(job))
        result.append({
            'id': str(job['_id']),
            'title': job['TITLE'],
            'company': job['COMPANY'],
            'location': job['LOCATION'],
            'job_url': job['Job URL'],
        })
    print(result)
    return result


def get_score(file):
    # Read the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    with open(file_path, 'r', encoding='ISO-8859-1') as f:
        text = f.read()
    os.remove(file_path)
    # Count the number of words
    word_count = len(re.findall(r'\b\w+\b', text))

    # Define the list of predefined words to match
    predefined_words = [
    'ability', 'abstract', 'academic', 'account', 'accountability', 'accounting', 'accuracy', 'achievement',
    'acquisition', 'action', 'activation', 'active', 'activity', 'adaptability', 'adept', 'administration',
    'administrative', 'admission', 'adoption', 'advertising', 'advising', 'advisory', 'advocacy', 'agility',
    'aid', 'alert', 'alignment', 'alleviation', 'alliance', 'allocation', 'allowance', 'alpha', 'analysis',
    'analytical', 'analytics', 'anatomy', 'and', 'animation', 'anonymity', 'anticipation', 'appraisal',
    'appreciation', 'approach', 'approval', 'architect', 'architecture', 'archives', 'area', 'arrangement',
    'articulation', 'artificial', 'aspiration', 'assessment', 'asset', 'assignment', 'assimilation',
    'assistance', 'associate', 'assurance', 'attractiveness', 'audit', 'authentication', 'authoring',
    'authority', 'automation', 'autonomy', 'availability', 'awareness', 'backlog', 'balance', 'banking',
    'behavior', 'beneficiary', 'benefit', 'beta', 'biography', 'biometrics', 'biotechnology', 'bloc',
    'blogging', 'board', 'bookkeeping', 'boost', 'boundary', 'brand', 'brevity', 'budget', 'building',
    'bulk', 'business', 'calendar', 'call', 'campaign', 'capability', 'capacity', 'capital', 'care',
    'career', 'case', 'catalog', 'category', 'cause', 'cell', 'center', 'certainty', 'certification',
    'challenge', 'change', 'character', 'chart', 'checklist', 'chemistry', 'chief', 'childcare',
    'circulation', 'circumstance', 'citizenship', 'city', 'civilization', 'claim', 'class', 'classification',
    'cleaning', 'clearance', 'client', 'climbing', 'cognition', 'collaboration', 'colleague', 'collection',
    'college', 'color', 'commerce', 'commercial', 'commitment', 'communication', 'community', 'company',
    'compensation', 'competence', 'competition', 'compliance', 'composition', 'comprehension', 'compression',
    'computing', 'concept', 'concentration', 'conceptual', 'conclusion', 'condensation', 'condition',
    'conduct', 'confidentiality', 'configuration', 'confirmation', 'conflict', 'conformance', 'congress',
    'connection', 'consciousness', 'consent', 'consequence', 'conservation', 'consideration', 'consistency',
    'consolidation', 'constituency', 'construction', 'consultant', 'consultation', 'consumer', 'consumption',
    'contact', 'containment', 'content', 'context', 'continuity', 'contract', 'contribution', 'control',
    'convenience', 'conversation', 'conversion', 'cooperation', 'coordination', 'corporation', 'correctness',
    'cost', 'counsel', 'countermeasure', 'country', 'coverage', 'creation', 'credibility', 'credit',
    'crime', 'criticism', 'culture', 'data', 'database','debugging',
    'decision-making', 'delivering', 'deployment', 'design', 'detail-oriented',
    'developed', 'developing', 'development', 'diagnosing', 'digital',
    'directing', 'distributed', 'documentation', 'drafting', 'driven',
    'dynamic', 'e-commerce', 'earned', 'easily', 'ecosystems', 'editing',
    'education', 'effective', 'efficiency', 'electronic', 'elements',
    'emerging', 'empowerment', 'enabled', 'encompassing', 'encouraged',
    'end-to-end', 'energized', 'engineered', 'enhanced', 'enjoyed', 'enriched',
    'enterprise', 'enthusiastic', 'entrepreneurial', 'environment', 'equity',
    'error-free', 'escalation', 'established', 'estimation', 'ethics',
    'evaluating', 'event', 'evidence-based', 'evolution', 'exceeding',
    'excelled', 'excellence', 'exceptional', 'excessive', 'exchange',
    'execution', 'executive', 'exemplary', 'exercising', 'exhibiting',
    'experienced', 'expertise', 'exploring', 'exporting', 'expressing',
    'extensive', 'extraordinary', 'fabrication', 'facilitation', 'familiar',
    'fashioned', 'feasibility', 'feature', 'feedback', 'fields', 'financial',
    'fired', 'fiscal', 'fitted', 'flexibility', 'focused', 'focusing',
    'followed', 'forecasting', 'foreign', 'formalized', 'formulated',
    'forward-thinking', 'fostering', 'frameworks', 'free-flowing', 'frequent',
    'friendly', 'front-end', 'full-stack', 'functionality', 'fundamentals',
    'funded', 'furthering', 'fusion', 'future-proof', 'gain',
    'game-changing', 'gave', 'generating', 'generation', 'geospatial',
    'germane', 'gleaned', 'global', 'go-to-market', 'goal-oriented', 'good',
    'governance', 'granted', 'graphical', 'grasped', 'ground-up', 'group',
    'growth', 'guided', 'handling', 'harnessed', 'head', 'healthcare', 'helped',
    'high-quality', 'high-volume', 'higher-level', 'holistic', 'homogeneous',
    'hosted', 'hourly', 'human', 'hybrid', 'hyperscale', 'identified',
    'identifying', 'ignited', 'illustrated', 'immaculate', 'implemented',
    'implied', 'important', 'improved', 'improving', 'in-bound', 'incentivized',
    'included', 'improve', 'implement', 'innovate', 'inspire', 'initiative', 
    'interact', 'investigate', 'inventory', 'issue', 'integrate', 'integrity', 
    'intensive', 'instruct', 'interpret', 'inform', 'involve', 'influence', 'identify', 
    'ideate', 'interview', 'impress', 'international', 'important', 'innovative', 
    'interpersonal', 'independent', 'inclusive', 'impactful', 'insightful', 'ideal', 
    'informed', 'integral', 'interesting', 'initiate', 'illustrate', 'invaluable', 
    'inquisitive', 'incorporate', 'innovation', 'insight', 'inspired', 'impressive', 
    'innovating', 'intentional', 'investigating', 'improving', 'influencing', 
    'integrating', 'inspiring', 'involving', 'implementing', 'java',
    'javascript','jenkins','job','join','journal','journalism','journalist','jquery','js','json',
    'junit', 'kanban', 'kendo', 'keynote', 'keyword', 'kindness', 'knowledge', 'known', 'kpi', 'kubernetes',
    'languages', 'leadership', 'learn', 'learning', 'legal', 'liaison', 'license',     'linkedin', 
    'linux', 'listening', 'literacy', 'logistics', 'long-term', 'loyalty',
    'machine learning', 'maintenance', 'management', 'manager', 'manufacturing', 
    'market analysis', 'marketing', 'mathematics', 'matlab', 'mechanical', 'media', 
    'medicine', 'meeting deadlines', 'mentoring', 'microsoft access', 'microsoft excel', 
    'microsoft office', 'microsoft outlook', 'microsoft powerpoint', 'microsoft word', 
    'mobile', 'modeling', 'mongodb', 'motivation', 'ms excel', 'ms office', 'ms outlook', 
    'ms powerpoint', 'ms project', 'ms word', 'multi-tasking', 'mysql',
    'narrate', 'navigate', 'negotiate', 'network', 'new', 'nominate', 'normalize', 
    'note', 'notice', 'notify', 'novel', 'now', 'nurture', 'nutrition', 'nutritional',
    'objective', 'observational', 'obtained', 'occupational', 'occur', 'offer', 
    'offered', 'offering',     'offers', 'office', 'officer', 'official', 'often', 
    'omitted', 'on-site', 'onboard', 'ongoing',     'online', 'onsite', 'onwards', 
    'open', 'opened', 'opening', 'operations', 'opportunities',     'opportunity', 
    'optimization', 'optimizing', 'oral', 'order', 'organized', 'original', 'other',     
    'outcomes', 'outline', 'outlined', 'outlining', 'outlook', 'outstanding', 'overcame', 
    'oversee',     'overseeing', 'oversaw', 'oversees', 'own', 'owned', 'owner', 'ownership',
    'packaging', 'pallet', 'paperwork', 'participate', 'partnership',
    'payment', 'peer', 'performance', 'permit', 'personable',
    'personalize', 'pharmaceutical', 'pharmacy', 'phone', 'photo',
    'photography', 'photoshop', 'physician', 'pipeline', 'plan',
    'planning', 'plc', 'pleasant', 'plus', 'policies', 'policy',
    'polish', 'pop', 'portfolio', 'position', 'positive',
    'possess', 'post', 'poster', 'postings', 'potential',
    'powerpoint', 'practices', 'practitioner', 'pre', 'preparation',
    'prepare', 'present', 'presentation', 'presentations', 'preserve',
    'press', 'prevent', 'preventive', 'pricing', 'print',
    'printers', 'printing', 'prioritization', 'prioritize', 'private',
    'problem', 'problem-solving', 'procedures', 'process',
    'processing', 'proctor', 'produce', 'production', 'productive',
    'productivity', 'professional', 'proficiency', 'profile', 'profit',
    'program', 'programming', 'project', 'promote', 'promotions',
    'proofread', 'property', 'proposals', 'prospects', 'protection',
    'protocol', 'prototype', 'proven', 'provide', 'provider',
    'providing', 'ps', 'psychology', 'public', 'publication',
    'publicity', 'purchase', 'purchasing', 'quality', 'quantitative',
    'queries', 'question', 'quickbooks', 'quota', 'quote', 'r',
    'react', 'read', 'reading', 'real', 'reception', 'receptionist',
    'recommend', 'reconcile', 'reconciliation', 'record', 'recruit',
    'recruiting', 'recruitment', 'reduce', 'refer', 'reference',
    'referral', 'referred', 'regard', 'register', 'regulation',
    'rehabilitation', 'reimbursement', 'related', 'relationship',
    'relationships', 'release', 'relevant', 'reliability', 'reliable',
    'rely', 'remain', 'remarkable', 'remote', 'repair', 'repeat', 'replenishment', 
    'report', 'reporting', 'reports', 'represent', 'representative', 'request', 'require',
    'required', 'requirements', 'research', 'reserve', 'residential', 'residents', 'residual',
    'resolution', 'resolve', 'resort', 'resources', 'respectful', 'responsibilities',
    'responsibility', 'responsible', 'responsive', 'rest', 'restaurant', 'restore',
    'result', 'results', 'resume', 'retail',
    'sales',    'salary',    'schedule',    'scope',    'screening',    'scripting',
    'search',    'security',    'self-motivated',    'sensitivity',    'service',
    'skills',    'software',    'solution',    'source',    'specifications',    
    'staff',    'statistics',    'strategy',    'streamlining',    'structure',    
    'style',    'subcontractors',    'submissions',    'success',    'suggestions',    
    'supervision',    'support',    'system',    'table',    'tasks',    'tax',    'team',    
    'technologies',    'testing',    'time',    'timelines',    'tools',    'top-performing',   
    'training',    'transparency',    'transportation',    'troubleshooting',    'typeface',    
    'typography',    'university',    'up-to-date',    'updating',    'usability',    'user',    
    'utilities',    'utilizing',    'validation',    'values',    'vendors',    'verification',    
    'versions',    'video',    'virtual',    'vision',    'visualization',    'voice',    'volume',    
    'web',    'website',    'willingness',    'workflow',    'workload',    'writing',
    'x-ray',  'xcode',  'xml',  'yaml',  'yearly',  'yield',  'yml',  'ytd',  'zeal',  
    'zero-defect',  'zero-injury',  'zero-tolerance',  'zoning',  'zookeeper',  'zoom',  'zsh',
    'education' , 'b-tech', 'tech', 'highschool', 'gpa', 'grade', 'cgpa', 'contest', 'rank',
    'Google', 'Microsoft', 'Apple', 'Amazon', 'Facebook', 'IBM', 'Intel', 'Oracle', 'Cisco', 
    'HP', 'Dell', 'VMware', 'Salesforce', 'Adobe', 'Symantec', 'PayPal', 'eBay', 'Twitter', 
    'LinkedIn', 'Snap', 'Uber', 'Airbnb', 'Netflix', 'Tesla', 'SpaceX', 'Qualcomm', 'Nvidia', 
    'AMD', 'Samsung', 'LG', 'Sony', 'Nintendo', 'Canon', 'Nokia', 'Ericsson', 'Huawei', 'Alibaba', 
    'Tencent', 'Baidu', 'Xiaomi', 'JD.com', 'ByteDance', 'TikTok',
    'Python', 'Java', 'Machine Learning', 'Data Analysis', 'C++', 'MySQL', 'Algorithms',
    'JavaScript', 'PHP', 'HTML', 'CSS', 'React', 'Angular', 'Vue.js', 'Node.js', 
    'Express.js', 'MongoDB', 'SQL', 'PostgreSQL', 'Git', 'GitHub', 'Docker', 
    'Kubernetes', 'AWS', 'Azure', 'Google Cloud Platform', 'Linux', 'Windows', 
    'MacOS', 'Android', 'iOS', 'Flutter', 'Django', 'Flask', 'PyTorch', 'TensorFlow', 
    'Keras', 'Numpy', 'Pandas', 'Matplotlib', 'Seaborn', 'Scikit-learn', 'NLTK', 
    'OpenCV', 'Unity', 'Unreal Engine', 'Blender', 'Maya', 'Photoshop', 'Illustrator', 
    'InDesign', 'Premiere Pro', 'After Effects', 'Final Cut Pro', 'Logic Pro', 'Pro Tools', 
    'Ableton Live', 'AutoCAD', 'Revit', 'SketchUp', 'SolidWorks', 'R', 'Scala', 'Hadoop', 
    'Spark', 'Hive', 'Pig', 'MapReduce', 'Kafka', 'Zookeeper', 'Flume', 'Sqoop', 'Impala', 
    'Oozie', 'Nifi', 'Tableau', 'Power BI', 'Excel', 'Google Sheets', 'Project Management', 
    'Agile', 'Scrum', 'Waterfall', 'Jira', 'Confluence', 'Trello', 'Asana', 'Slack', 
    'Microsoft Office', 'Google Workspace', 'Salesforce', 'Zendesk', 'HubSpot', 
    'Mailchimp', 'AdWords', 'Analytics', 'SEO', 'SEM', 'Content Marketing', 
    'Social Media Marketing', 'Email Marketing', 'Copywriting', 'Editing', 
    'Proofreading', 'Technical Writing', 'Creative Writing', 'UX Design', 'UI Design', 
    'Graphic Design', 'Web Design', 'Mobile Design', 'Product Design', 
    'Product Management', 'Customer Service', 'Leadership', 'Teamwork', 
    'Communication', 'Problem Solving', 'Critical Thinking', 'Time Management', 
    'Organizational Skills', 'Multitasking', 'Negotiation', 'Sales', 'Marketing', 
    'Business Development', 'Finance', 'Accounting', 'Human Resources', 
    'Recruiting', 'Training', 'Coaching', 'Mentoring', 'Public Speaking', 
    'Event Planning', 'Fundraising', 'Nonprofit Management', 'Grant Writing', 
    'Volunteer Management', 'Project Coordination', 'Logistics', 'Supply Chain Management', 
    'Quality Assurance', 'Quality Control', 'Risk Management', 'Compliance', 
    'Legal', 'Contract Management', 'Intellectual Property', 'Patents', 
    'Trademarks', 'Privacy', 'Security', 'Cybersecurity', 'Ethical Hacking', 
    'Penetration Testing', 'IT Support', 'Network Administration', 'System Administration', 
    'Database Administration', 'Software Testing', 'QA', 'DevOps', 
    'Site Reliability Engineering', 'Cloud Architecture', 'Virtualization', 'Automation', 
    'Scripting', 'Bash', 'Python Scripting', 'PowerShell', 'Ruby', 'Perl', 'C#', 'Swift', 
    'Objective-C', 'Go', 'Rust', 'Scala'
    ]


    # Count the number of matches
    match_count = 0
    for word in predefined_words:
        match_count += len(re.findall(word, text, re.IGNORECASE))
        print(match_count)

    # Calculate the score as a percentage
    score = (match_count*5 / word_count) * 100

    return round(score)



@app.route('/jobs/<string:role>/<string:location>')
def get_jobs(role, location):
    jobs = mongo.db.JOBS_Basic.find({
        'TITLE': {'$regex': f'.*{role}.*', '$options': 'i'},
        'LOCATION': {'$regex': f'.*{location}.*', '$options': 'i'}
    })
    result = []
    for job in jobs:
        result.append({
            'id': str(job['_id']),
            'title': job['TITLE'],
            'company': job['COMPANY'],
            'location': job['LOCATION'],
            'job_url': job['Job URL'],
        })
        print(result)
    return jsonify(result)

        
@app.route('/jobs/search', methods=['POST'])
def search_jobs():
    file = request.files['file']
    # process the file to get matching jobs
    jobs = process_file(file)
    return jsonify(jobs)


    
@app.route('/jobs/analyse', methods=['POST'])
def analyse_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    # Get the file from the request
    file = request.files['file']

    # Calculate the score
    score = get_score(file)

    # Return the score as JSON
    return jsonify({'score': score})

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
