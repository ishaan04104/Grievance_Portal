import re
import spacy
from flask import Flask, render_template, request, redirect, url_for, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from transformers import pipeline, AutoTokenizer, BertForSequenceClassification
import nltk
from datetime import datetime, timedelta
import pytz  # Import pytz for timezone handling

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

# Load the summarization pipeline and tokenizer using BART
model_name = "facebook/bart-large-cnn"  # Using BART for summarization
summarizer = pipeline("summarization", model=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the sentiment analysis model (BERT)
sentiment_model_name = "nlptown/bert-base-multilingual-uncased-sentiment"  # A BERT model for sentiment analysis
sentiment_model = BertForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_model_name)

# Initialize Flask app and database
app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session storage
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///journal.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
migrate = Migrate(app, db)  # Initialize Flask-Migrate

# Define the JournalEntry model
class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    summary = db.Column(db.Text, nullable=True)
    sentiment = db.Column(db.Text, nullable=True)  # Changed from topics to sentiment
    date_created = db.Column(db.DateTime, default=datetime.now(pytz.timezone('Asia/Kolkata')))  # Set default to IST

    def __repr__(self):
        return f'<JournalEntry {self.id}>'

def clean_text(text):
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

def chunk_text(text, max_length=256):
    """Chunk text into smaller segments of max_length tokens."""
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []
    
    for i in range(0, len(tokens), max_length):
        chunk = tokens[i:i + max_length]
        chunks.append(tokenizer.decode(chunk, skip_special_tokens=True))
    
    return chunks

def summarize_journal_entry(entry):
    # Chunk the entry and summarize each chunk
    chunks = chunk_text(entry)
    summaries = []
    
    for chunk in chunks:
        summary = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    
    return " ".join(summaries)  # Combine summaries from all chunks

def analyze_sentiment(text):
    # Analyze sentiment for the entire text
    sentiment = sentiment_analyzer(text)
    
    # Extract the sentiment label
    sentiment_label = sentiment[0]['label']  # This will give you the sentiment label

    # Map the sentiment labels to user-friendly terms
    if sentiment_label in ['5 stars', '4 stars']:
        return 'happy', sentiment[0]['score']
    elif sentiment_label == '3 stars':
        return 'neutral', sentiment[0]['score']
    else:  # For LABEL_1 and LABEL_2
        return 'sad', sentiment[0]['score']

@app.route('/')
def home():
    return redirect(url_for('express_feelings'))

@app.route('/express_feelings', methods=['GET', 'POST'])
def express_feelings():
    if request.method == 'POST':
        content = request.form['content']
        
        # Process the content and generate summary
        summary = summarize_journal_entry(content)
        
        # Analyze sentiment
        sentiment_label, sentiment_score = analyze_sentiment(content)
        
        # Save the journal entry with summary and sentiment
        new_entry = JournalEntry(content=content, summary=summary, sentiment=sentiment_label)
        db.session.add(new_entry)
        db.session.commit()
        
        return redirect(url_for('express_feelings'))

    return render_template('express_feelings.html')

@app.route('/journals')
def journals():
    sentiment_filter = request.args.get('sentiment')
    date_filter = request.args.get('date')

    # Get all entries
    entries = JournalEntry.query

    # Filter by sentiment if specified
    if sentiment_filter:
        entries = entries.filter(JournalEntry.sentiment == sentiment_filter)

    # Sort by date if specified
    if date_filter == 'week':
        entries = entries.filter(JournalEntry.date_created >= (datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(weeks=1)))
    elif date_filter == 'month':
        entries = entries.filter(JournalEntry.date_created >= (datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(days=30)))
    elif date_filter == 'year':
        entries = entries.filter(JournalEntry.date_created >= (datetime.now(pytz.timezone('Asia/Kolkata')) - timedelta(days=365)))

    entries = entries.order_by(JournalEntry.date_created.desc()).all()
    return render_template('journals.html', entries=entries)

@app.route('/delete_entry/<int:entry_id>', methods=['POST'])
def delete_entry(entry_id):
    entry = JournalEntry.query.get_or_404(entry_id)
    db.session.delete(entry)  # This deletes both the content and the summary
    db.session.commit()
    return redirect(url_for('journals'))

@app.route('/summaries')
def summaries():
    entries = JournalEntry.query.order_by(JournalEntry.date_created.desc()).all()
    return render_template('summaries.html', entries=entries)

@app.route('/clear')
def clear_entries():
    db.session.query(JournalEntry).delete()  # Clear all entries
    db.session.commit()
    return redirect(url_for('journals'))

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        user_input = request.json.get('message')
        bot_response = generate_conversational_response(user_input)
        return jsonify({'response': bot_response})

    return render_template('chatbot.html')

def generate_conversational_response(user_input):
    if "chat_history" not in session:
        session["chat_history"] = []  # Initialize history if it's empty

    # Fetch the last two journal summaries
    last_entries = JournalEntry.query.order_by(JournalEntry.date_created.desc()).limit(2).all()
    context = " ".join([entry.summary for entry in last_entries if entry.summary])  # Combine summaries

    # Append user message to chat history
    session["chat_history"].append(f"You: {user_input}")

    # Create a structured prompt
    full_conversation = (
        "You are a compassionate and insightful counselor, trained to provide human-like therapeutic guidance. "
        "The person speaking to you is going through an emotionally difficult time. Your role is to offer warmth, understanding, and a safe space for them to open up.\n\n"
        "They have shared summaries of their last two journal entries. Instead of immediately offering solutions, start by validating their emotions, expressing empathy, and encouraging them to elaborate.\n\n"
        "Let the conversation feel natural, allowing them to guide the discussion. Use open-ended questions and gentle prompts to help them reflect and share more.\n\n"
        "As they open up, gradually introduce insights and coping strategies without overwhelming them. Make the conversation feel like a supportive dialogue rather than an advice dump.\n\n"
        "Once the user has expressed themselves sufficiently:\n"
        "- Acknowledge and summarize their feelings in a comforting way.\n"
        "- Offer encouragement and remind them of their strengths.\n"
        "- Suggest one actionable next step that feels achievable for them.\n"
        "- End the conversation with warmth and an open invitation to return whenever they need support.\n\n"
        "Journal Summaries:\n"
        f"{context}\n\n"
        "Now, based on the above context, respond to the following:\n"
        f"{user_input}\n\n"
        "Therapeutic Response:\n"
    )

    try:
        response = model.generate_content(full_conversation)
        bot_reply = response.text.strip() if response.text else "I didn't understand that."

        # Store bot reply in chat history
        session["chat_history"].append(f"Bot: {bot_reply}")

        return bot_reply
    except Exception as e:
        print("Error:", e)
        return "Oops! Something went wrong."

if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables
    app.run(debug=True, port=3550)