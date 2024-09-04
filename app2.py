import streamlit as st
import pickle
import pandas as pd
import hashlib
import json
import time
from datetime import datetime
import numpy as np

# Define the blockchain classes
import streamlit as st
import pickle
import pandas as pd
import hashlib
import json
import time
from datetime import datetime
import numpy as np

# Define the blockchain classes
class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        # Check if self.data is a dictionary
        if isinstance(self.data, dict):
            # Convert NumPy int64 values to regular Python integers
            data = {k: int(v) if isinstance(v, np.int64) else v for k, v in self.data.items()}
        else:
            data = self.data

        block_string = json.dumps({"index": self.index, "timestamp": self.timestamp, "data": data, "previous_hash": self.previous_hash}, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), "Genesis Block", "0")
        self.chain.append(genesis_block)

    def add_block(self, block):
        block.previous_hash = self.chain[-1].hash
        block.hash = block.compute_hash()
        self.chain.append(block)

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.compute_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

# Load the pre-trained models
models = {
    'SVM': 'models/svc.pkl',
    'XGBoost': 'models/xgb.pkl',
    'Decision Tree': 'models/clf.pkl'
}

# Define columns for the new dataset
columns = ['Student_ID', 'Timestamp', 'SSLC', 'HSC', 'CGPA', 'School_Type', 'No_of_Miniprojects', 'No_of_Projects', 'Coresub_Skill', 'Aptitude_Skill', 'Problemsolving_Skill', 'Programming_Skill', 'Abstractthink_Skill', 'Design_Skill', 'First_Computer', 'First_Program', 'Lab_Programs', 'DS_Coding', 'Technology_Used', 'Sympos_Attend', 'Sympos_Won', 'Extracurricular', 'Learning_Style', 'College_Bench', 'Clg_Teachers_Know', 'College_Performence', 'College_Skills', 'Prediction']

st.title('Prediction')

# Define the input fields
# Select the model
model_name = st.selectbox('Select Model', list(models.keys()))

# Get student ID from user input
student_id = st.text_input("Enter Student ID")

# Get student data from user input
sslc = st.number_input('SSLC', min_value=0.0, max_value=100.0, value=0.0)
hsc = st.number_input('HSC', min_value=0.0, max_value=100.0, value=0.0)
cgpa = st.number_input('CGPA', min_value=0.0, max_value=10.0, value=0.0)
school_type = st.number_input('School_Type', min_value=0, max_value=2, value=0)
no_of_miniprojects = st.number_input('No_of_Miniprojects', min_value=0, step=1, value=0)
no_of_projects = st.number_input('No_of_Projects', min_value=0, step=1, value=0)
coresub_skill = st.number_input('Coresub_Skill', min_value=0.0, max_value=10.0, value=0.0)
aptitude_skill = st.number_input('Aptitude_Skill', min_value=0.0, max_value=10.0, value=0.0)
problemsolving_skill = st.number_input('Problemsolving_Skill', min_value=0.0, max_value=10.0, value=0.0)
programming_skill = st.number_input('Programming_Skill', min_value=0.0, max_value=10.0, value=0.0)
abstractthink_skill = st.number_input('Abstractthink_Skill', min_value=0.0, max_value=10.0, value=0.0)
design_skill = st.number_input('Design_Skill', min_value=0.0, max_value=10.0, value=0.0)
first_computer = st.number_input('First_Computer', min_value=0, max_value=2, value=0)
first_program = st.number_input('First_Program', min_value=0, max_value=2, value=0)
lab_programs = st.number_input('Lab_Programs', min_value=0, max_value=2, value=0)
ds_coding = st.number_input('DS_Coding', min_value=0, max_value=2, value=0)
technology_used = st.number_input('Technology_Used', min_value=0, max_value=2, value=0)
sympos_attend = st.number_input('Sympos_Attend', min_value=0, step=1, value=0)
sympos_won = st.number_input('Sympos_Won', min_value=0, step=1, value=0)
extracurricular = st.number_input('Extracurricular', min_value=0, max_value=2, value=0)
learning_style = st.number_input('Learning_Style', min_value=0, max_value=2, value=0)
college_bench = st.number_input('College_Bench', min_value=0, max_value=2, value=0)
clg_teachers_know = st.number_input('Clg_Teachers_Know', min_value=0, max_value=2, value=0)
college_performence = st.number_input('College_Performence', min_value=0.0, max_value=10.0, value=0.0)
college_skills = st.number_input('College_Skills', min_value=0.0, max_value=10.0, value=0.0)

blockchain = Blockchain()

# Create a button to trigger the prediction and blockchain addition
if st.button('Predict & Add to Blockchain'):
    # Prepare the input data
    with open(models[model_name], 'rb') as file:
        model = pickle.load(file)
    input_data = [[sslc, hsc, cgpa, school_type, no_of_miniprojects, no_of_projects, coresub_skill, aptitude_skill,
                   problemsolving_skill, programming_skill, abstractthink_skill, design_skill, first_computer,
                   first_program, lab_programs, ds_coding, technology_used, sympos_attend, sympos_won,
                   extracurricular, learning_style, college_bench, clg_teachers_know, college_performence,
                   college_skills]]

    # Make the prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'Predicted Output: {prediction[0]}')

    # Add data to the blockchain
    block_data = {
        'Student_ID': student_id,
        'Timestamp': time.time(),
        'SSLC': sslc,
        'HSC': hsc,
        'CGPA': cgpa,
        'School_Type': school_type,
        'No_of_Miniprojects': no_of_miniprojects,
        'No_of_Projects': no_of_projects,
        'Coresub_Skill': coresub_skill,
        'Aptitude_Skill': aptitude_skill,
        'Problemsolving_Skill': problemsolving_skill,
        'Programming_Skill': programming_skill,
        'Abstractthink_Skill': abstractthink_skill,
        'Design_Skill': design_skill,
        'First_Computer': first_computer,
        'First_Program': first_program,
        'Lab_Programs': lab_programs,
        'DS_Coding': ds_coding,
        'Technology_Used': technology_used,
        'Sympos_Attend': sympos_attend,
        'Sympos_Won': sympos_won,
        'Extracurricular': extracurricular,
        'Learning_Style': learning_style,
        'College_Bench': college_bench,
        'Clg_Teachers_Know': clg_teachers_know,
        'College_Performence': college_performence,
        'College_Skills': college_skills,
        'Prediction': prediction[0]
    }

    # Add the data to the blockchain
    new_block = Block(len(blockchain.chain), time.time(), block_data, blockchain.chain[-1].hash)
    blockchain.add_block(new_block)

    # Display blockchain data
    # st.write("\nBlockchain:")
    # for block in blockchain.chain:
    #     st.write(f"Index: {block.index}")
    #     st.write(f"Timestamp: {datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
    #     st.write("Data:", block.data)
    #     st.write("Previous Hash:", block.previous_hash)
    #     st.write("Hash:", block.hash)
    #     st.write("")
    
    # Save the data to CSV file
    new_data = pd.DataFrame([block_data], columns=columns)
    with open("updated.csv", "a") as f:
        new_data.to_csv(f, header=f.tell()==0, index=False)
    
    # Show the data in the CSV file
    st.write("Data in the CSV file:")
    csv_data = pd.read_csv("updated.csv")
    st.write(csv_data)


