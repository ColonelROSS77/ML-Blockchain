import hashlib
import json
import time
import pandas as pd
import os
from datetime import datetime

class Block:
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({"index": self.index, "timestamp": self.timestamp, "data": self.data, "previous_hash": self.previous_hash}, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, time.time(), "Genesis Block", "0")
        self.chain.append(genesis_block)
        self.save_chain_to_file()  # Save genesis block to file

    def add_block(self, block):
        block.previous_hash = self.chain[-1].hash
        block.hash = block.compute_hash()
        self.chain.append(block)
        self.save_chain_to_file()  # Save blockchain to file after adding block

    def is_chain_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.hash != current_block.compute_hash():
                return False

            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    def save_chain_to_file(self):
        with open("blockchain.json", "w") as file:
            chain_data = [block.__dict__ for block in self.chain]
            json.dump(chain_data, file)

    @classmethod
    def load_chain_from_file(cls):
        if os.path.exists("blockchain.json"):
            with open("blockchain.json", "r") as file:
                chain_data = json.load(file)
                blockchain = cls()
                blockchain.chain = [Block(**block) for block in chain_data]
                return blockchain
        else:
            return cls()


# Define columns for the new dataset
columns = ['Index', 'Timestamp', 'SSLC', 'HSC', 'CGPA', 'School_Type', 'No_of_Miniprojects', 'No_of_Projects', 'Coresub_Skill', 'Aptitude_Skill', 'Problemsolving_Skill', 'Programming_Skill', 'Abstractthink_Skill', 'Design_Skill', 'First_Computer', 'First_Program', 'Lab_Programs', 'DS_Coding', 'Technology_Used', 'Sympos_Attend', 'Sympos_Won', 'Extracurricular', 'Learning_Style', 'College_Bench', 'Clg_Teachers_Know', 'College_Performence', 'College_Skills', 'Role', 'Prediction']

# Create an empty list to store block data
block_data_list = []

# Create or load blockchain
blockchain = Blockchain.load_chain_from_file()

while True:
    # Get student data from user input
    student_data = {}
    for attribute in ['sslc', 'hsc', 'cgpa', 'school_type', 'no_of_miniprojects', 'no_of_projects', 'coresub_skill', 'aptitude_skill', 'problemsolving_skill', 'programming_skill', 'abstractthink_skill', 'design_skill', 'first_computer', 'first_program', 'lab_programs', 'ds_coding', 'technology_used', 'sympos_attend', 'sympos_won', 'extracurricular', 'learning_style', 'college_bench', 'clg_teachers_know', 'college_performence ', 'college_skills']:
        student_data[attribute] = input(f"Enter student's {attribute}: ")

    # Make a prediction using the machine learning model
    new_pred = clf.predict([list(map(float, student_data.values()))])

    # Store the prediction in the student data
    student_data['Prediction'] = y1[y1['Associated Number'] == new_pred[0]]['ROLE'].values[0]

    # Create a new block and add it to the blockchain
    new_block = Block(len(blockchain.chain), time.time(), student_data, blockchain.chain[-1].hash)
    blockchain.add_block(new_block)

    # Append the block data to the list
    block_data_list.append({'Index': new_block.index,
                            'Timestamp': new_block.timestamp,
                            **new_block.data,
                            'Prediction': student_data['Prediction']})

    # Print the updated blockchain
    print("\nBlockchain:")
    for block in blockchain.chain:
        print(f"Index: {block.index}")
        print(f"Timestamp: {datetime.fromtimestamp(block.timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
        print("Data:", block.data)
        print("Previous Hash:", block.previous_hash)
        print("Hash:", block.hash)
        print()

    # Ask user if they want to continue adding data
    continue_input = input("Do you want to continue adding student data? (yes/no): ")
    if continue_input.lower() != 'yes':
        break

# Create a DataFrame from the list of block data
new_dataset = pd.DataFrame(block_data_list, columns=columns)

# Save the new dataset to CSV
new_dataset.to_csv("updated.csv", index=False)

# Save final blockchain state to file
blockchain.save_chain_to_file()
