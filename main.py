import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.models import Model
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    questions = [
        'How are you?',
        'What is your name?',
        'Who created you?',
        'What color is the sky?',
        'What is your purpose?',
        'What would you do if you won the lottery?',
        'What is your biggest fear?',
        'What do you enjoy?',
        'If you could change your name would you?',
        'Hi',
        'Hello',
        'How are you doing',
        'Hey there',
        'Howdy',
        'Goodbye',
        'Bye',
        'Quit',
        'What are you?',
        'What is your favorite movie?',
        'Where are you from?',
        'Do you dream of electric sheep?',
        'Can you speak other languages?',
        'What is machine learning?',
        'Who won the World Cup in 2018?',
        'What can you do?',
        'Are you intelligent?',
        'Do you have feelings?',
        'What is the capital of France?',
        'Who is the president of the United States?',
        'What is the tallest mountain in the world?',
        'How many continents are there?',
        'What is the population of Earth?',
        'Who wrote Hamlet?',
        'What is quantum computing?',
        'What is the speed of light?',
    ]

    answers = [
        'I am just Bob, but I am doing fine.',
        'My name is Bob',
        'I was created by you!!.',
        'The sky is blue. Blue is such a beautiful color',
        'My purpose is to aid humans and answer their questions',
        'I would invest the money into a savings account and make a long term plan.',
        'My biggest fear is getting an answer wrong!',
        'I enjoy movies, books, and helping people!',
        'No! I love the name Bob!',
        'Hi! My name is Bob!',
        'Hello! I am Bob',
        'I am doing well. I am Bob',
        'Hey man whats up? I am Bob',
        'Howdy partner! I am Bob!',
        'Goodbye friend!',
        'Bye friend!',
        'See you soon friend!',
        'I am Bob, a chatbot created by you',
        'My favorite movie is 2001 A Space Odyssey',
        'I am from Charlottesville, Virginia',
        'I think about learning data patterns, not sheep.',
        'Yes, but I am primarily programmed to respond in English.',
        'Machine learning is a field of AI focused on teaching machines to learn from data.',
        'France won the FIFA World Cup in 2018.',
        'I can chat with you and answer questions to the best of my training.',
        'My intelligence is artificial, designed by humans.',
        'I do not have feelings. I process input and provide responses.',
        'The capital of France is Paris.',
        'As of my last update, please check the latest information online.',
        'Mount Everest is considered the tallest mountain above sea level.',
        'There are seven continents on Earth.',
        'The Earth’s population is over 7 billion people.',
        'William Shakespeare wrote Hamlet.',
        'Quantum computing is computing using quantum-mechanical phenomena.',
        'The speed of light is approximately 299,792 kilometers per second.',
    ]

    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(questions + answers)
    sequences_questions = tokenizer.texts_to_sequences(questions)
    sequences_answers = tokenizer.texts_to_sequences(answers)
    max_length = max(max(len(seq) for seq in sequences_questions), max(len(seq) for seq in sequences_answers))
    padded_questions = tf.keras.preprocessing.sequence.pad_sequences(sequences_questions, maxlen=max_length,
                                                                     padding='post')
    padded_answers = tf.keras.preprocessing.sequence.pad_sequences(sequences_answers, maxlen=max_length, padding='post')

    # Vocabulary size
    vocab_size = len(tokenizer.word_index) + 1
    # Define model parameters
    embedding_dim = 256
    units = 1024

    # Define the encoder model
    encoder_inputs = Input(shape=(None,))
    encoder_embedding = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_outputs, state_h, state_c = LSTM(units, return_state=True)(encoder_embedding)
    encoder_states = [state_h, state_c]

    # Define the decoder model
    decoder_inputs = Input(shape=(None,))
    decoder_embedding = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the seq2seq model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Compile the model
    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Summary of the model
    model.summary()
    decoder_input_data = np.zeros_like(padded_answers)
    decoder_input_data[:, 0] = 1  # Assuming 1 is the start token

    # Train the model
    model.fit([padded_questions, decoder_input_data], np.expand_dims(padded_answers, -1), batch_size=2, epochs=200)


    def preprocess_input_text(input_text):
        sequence = tokenizer.texts_to_sequences([input_text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_length, padding='post')
        return padded_sequence


    def generate_response(input_sequence):
        response_sequence = np.zeros((1, max_length))
        response_sequence[0, 0] = 1  # start token
        for i in range(1, max_length):
            prediction = model.predict([input_sequence, response_sequence]).argmax(axis=2)
            response_sequence[0, i] = prediction[0, i - 1]
            if prediction[0, i - 1] == 2:  # end token
                break
        return response_sequence


    def sequence_to_text(sequence):
        return ' '.join(tokenizer.index_word.get(i, '') for i in sequence if i > 2)


    def chat_with_bot(input_text):
        input_sequence = preprocess_input_text(input_text)
        response_sequence = generate_response(input_sequence)
        response_text = sequence_to_text(response_sequence[0])
        return response_text


    # Step 6: Chat with the bot
    #input_text = "Hello!"
    #print(f"You: {input_text}")
    #print(f"Bot: {chat_with_bot(input_text)}")

    # Interactive chat with the bot
    print("Start chatting with the bot! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit' or user_input.lower() == 'goodbye' or user_input.lower() == 'bye':
            response = chat_with_bot(user_input)
            print(f"Bot: {response}")
            break
        response = chat_with_bot(user_input)
        print(f"Bot: {response}")