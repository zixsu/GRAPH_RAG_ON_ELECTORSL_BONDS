import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from joblib import Parallel, delayed
from dateutil.parser import parse

class Model:
    def __init__(self, dataset_path):
        # Load dataset
        self.data = pd.read_csv('data/final_data.csv')

        # Convert date columns to datetime format
        self.data['Date of Encashment'] = pd.to_datetime(self.data['Date of Encashment'], errors='coerce')
        self.data['Date of Purchase'] = pd.to_datetime(self.data['Date of Purchase'], errors='coerce')

        # Convert denominations to float
        self.data['Denominations'] = self.data['Denominations'].astype(str).str.replace(',', '').astype(float)

        # Normalize names (lowercase & stripped)
        self.data['Name of the Political Party'] = self.data['Name of the Political Party'].str.strip().str.lower()
        self.data['Name of the Purchaser'] = self.data['Name of the Purchaser'].str.strip().str.lower()

        # Create a Knowledge Graph
        self.G = nx.Graph()
        for _, row in self.data.iterrows():
            party = row['Name of the Political Party']
            purchaser = row['Name of the Purchaser']
            bond = str(row['Bond Number'])
            amount = row['Denominations']

            self.G.add_node(party, type='party', label=party)
            self.G.add_node(purchaser, type='purchaser', label=purchaser)
            self.G.add_node(bond, type='bond', amount=amount, label=f"Bond {bond}")

            self.G.add_edge(party, bond, relation='encashed', color='green')
            self.G.add_edge(purchaser, bond, relation='purchased', color='blue')

        # Combine text for NLP processing
        self.data['text'] = (
            self.data['Name of the Political Party'] + " " +
            self.data['Name of the Purchaser'] + " " +
            self.data['Bond Number'].astype(str) + " " +
            self.data['Denominations'].astype(str) + " " +
            self.data['Date of Encashment'].dt.strftime('%Y-%m-%d') + " " +
            self.data['Date of Purchase'].dt.strftime('%Y-%m-%d')
        )

        # Initialize vectorizer and NLP model
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.X = self.vectorizer.fit_transform(self.data['text'])
        self.qa_pipeline = pipeline('question-answering', model='deepset/roberta-base-squad2', tokenizer='deepset/roberta-base-squad2')

    def format_indian(self, number):
        """Converts numbers to proper Indian comma formatting"""
        num_str = str(int(number))
        reversed_str = num_str[::-1]
        chunks = []
        if len(reversed_str) >= 3:
            chunks.append(reversed_str[:3][::-1])
        else:
            chunks.append(reversed_str[::-1])
        remaining = reversed_str[3:]
        for i in range(0, len(remaining), 2):
            chunk = remaining[i:i+2][::-1]
            chunks.append(chunk)
        return ','.join(reversed(chunks))

    def clean_date(self, date_str):
        """Cleans date strings by removing suffixes like 'st', 'nd', 'th', 'rd'."""
        return date_str.replace("nd", "").replace("th", "").replace("st", "").replace("rd", "").strip('?')

    def process_query(self, question):
        """Process user queries and return answers"""
        question_lower = question.lower()
        response = None

        # Case 1: Bond Number Direct Lookup
        if "bond number" in question_lower:
            bond_number = ''.join(filter(str.isdigit, question.split("bond number")[-1]))
            bond_data = self.data[self.data['Bond Number'].astype(str) == bond_number]

            if not bond_data.empty:
                if "party received" in question_lower:
                    response = bond_data['Name of the Political Party'].iloc[0].upper()
                elif "who purchased" in question_lower:
                    response = bond_data['Name of the Purchaser'].iloc[0].upper()
                elif "denomination" in question_lower:
                    amount = bond_data['Denominations'].iloc[0]
                    response = self.format_indian(amount)  # Indian formatting
                elif "purchased" in question_lower and "when" in question_lower:
                    response = bond_data['Date of Purchase'].dt.strftime('%d-%b-%y').iloc[0]
                elif "encashed" in question_lower:
                    response = bond_data['Date of Encashment'].dt.strftime('%d-%b-%y').iloc[0]
            else:
                response = "No data found for this bond number"

        # Case 2: Total amounts with date handling
        elif "total bond amount" in question_lower:
            date_str = None
            if "on" in question_lower:
                date_part = question_lower.split("on")[-1].strip()
                date_str = self.clean_date(date_part)

            if "encashed" in question_lower:
                party = question.split("by")[-1].split("on")[0].strip().lower()
                filtered = self.data[self.data['Name of the Political Party'] == party]
                if date_str:
                    try:
                        target_date = parse(date_str).date()
                        filtered = filtered[filtered['Date of Encashment'].dt.date == target_date]
                    except:
                        pass
                total = filtered['Denominations'].sum()
                response = f"{int(total)} or {int(total//100000)} Lakhs" if total > 0 else "No data"

            elif "purchased" in question_lower:
                purchaser = question.split("by")[-1].split("on")[0].strip().lower()
                filtered = self.data[self.data['Name of the Purchaser'] == purchaser]
                if date_str:
                    try:
                        target_date = parse(date_str).date()
                        filtered = filtered[self.data['Date of Purchase'].dt.date == target_date]
                    except:
                        pass
                total = filtered['Denominations'].sum()
                response = f"{int(total)} or {int(total//100000)} Lakhs" if total > 0 else "No data"

        # Case 3: Bond counts
        elif "total number of bonds" in question_lower:
            purchaser = question.split("by")[-1].split("on")[0].strip().lower()
            date_str = None
            if "on" in question_lower:
                date_part = question_lower.split("on")[-1].strip()
                date_str = self.clean_date(date_part)

            filtered = self.data[self.data['Name of the Purchaser'] == purchaser]
            if date_str:
                try:
                    target_date = parse(date_str).date()
                    filtered = filtered[filtered['Date of Purchase'].dt.date == target_date]
                except:
                    pass
            response = str(len(filtered))

        # Fallback to NLP model
        if not response:
            query_vec = self.vectorizer.transform([question])
            similarities = cosine_similarity(query_vec, self.X).flatten()
            top_indices = similarities.argsort()[-3:][::-1]
            bond_numbers = self.data.iloc[top_indices]['Bond Number'].astype(str).tolist()

            context = []
            for bond in bond_numbers:
                if bond in self.G.nodes:
                    neighbors = list(self.G.neighbors(bond))
                    for neighbor in neighbors:
                        if self.G.nodes[neighbor]['type'] == 'party':
                            context.append(f"{neighbor} encashed bond {bond}")
                        elif self.G.nodes[neighbor]['type'] == 'purchaser':
                            context.append(f"{neighbor} purchased bond {bond}")

            context = " ".join(context)

            # Use NLP-based QA model
            if context:
                result = self.qa_pipeline(question=question, context=context)
                response = result['answer']
            else:
                response = "Sorry, no relevant information found."

        return response
