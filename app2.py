from flask import Flask, request, jsonify, session, render_template_string
import google.generativeai as genai

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session storage

# Set up Gemini API key
genai.configure(api_key="AIzaSyA6EKKkJd8GSGt9hpYzBFxqL2AYIuB2bPU")
model = genai.GenerativeModel("gemini-pro")

# Function to generate a response while keeping chat history
def generate_conversational_response(user_input):
    if "chat_history" not in session:
        session["chat_history"] = []  # Initialize history if it's empty

    # Append user message to chat history
    session["chat_history"].append(f"You: {user_input}")

    # Combine history for context-aware response
    full_conversation = "\n".join(session["chat_history"])
    
    try:
        response = model.generate_content(full_conversation)
        bot_reply = response.text.strip() if response.text else "I didn't understand that."

        # Store bot reply in chat history
        session["chat_history"].append(f"Bot: {bot_reply}")

        return bot_reply
    except Exception as e:
        print("Error:", e)
        return "Oops! Something went wrong."

# HTML Interface for Chat
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Gemini Chatbot</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 50px; }
        #chat-box { width: 100%; height: 300px; border: 1px solid black; padding: 10px; overflow-y: scroll; }
        #user-input { width: 80%; padding: 5px; }
        button { padding: 5px; }
    </style>
</head>
<body>
    <h1>Gemini Chatbot</h1>
    <div id="chat-box"></div>
    <form action="/ask" method="POST">
        <input type="text" id="user-input" name="question" placeholder="Say something..." />
        <button type="submit">Send</button>
    </form>

    <script>
        // Keep the chatbox scrolling to the bottom
        window.onload = function() {
            var chatBox = document.getElementById("chat-box");
            chatBox.scrollTop = chatBox.scrollHeight;
        };
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.form['question']
    bot_response = generate_conversational_response(user_question)
    
    # Display conversation in the chat box
    return render_template_string(HTML_TEMPLATE + '''
        <script>
            var chatBox = document.getElementById("chat-box");
            chatBox.innerHTML += "<p><strong>You:</strong> " + "{{ user_question }}" + "</p>";
            chatBox.innerHTML += "<p><strong>Bot:</strong> " + "{{ bot_response }}" + "</p>";
            chatBox.scrollTop = chatBox.scrollHeight;
        </script>
    ''', user_question=user_question, bot_response=bot_response)

if __name__ == '__main__':
    app.run(debug=True)
