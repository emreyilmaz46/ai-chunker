import nltk
import instructor
from instructor import Mode
from openai import OpenAI
from groq import Groq
from anthropic import Anthropic
import os
from dotenv import load_dotenv
from datamodels import EnhancedChunk, TextChunks

nltk.download('punkt')

load_dotenv()

client = instructor.patch(OpenAI(api_key=os.getenv("OPENAI_API_KEY")), mode=Mode.JSON)

#Workign with Groq - don't forget to to adjust the model name as well
# groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
# client = instructor.from_groq(groq_client, mode=instructor.Mode.TOOLS)
#Unworking trial for Anthropic models
# anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# client = instructor.from_anthropic(anthropic_client)


def execute_llm_chunking(model, response_model, messages):

    AI_Response = client.chat.completions.create(
        model=model,
        response_model=response_model,
        messages=messages,
        temperature=0
    )

    return AI_Response


def process_text(input_text):
    # Tokenize the text into sentences
    sentences = nltk.sent_tokenize(input_text)

    # Insert artifacts after each sentence
    text_with_artifacts = ""
    for i, sentence in enumerate(sentences):
        text_with_artifacts += f"{sentence} [{i}]\n"

    # Use OpenAI to determine chunk boundaries with Instructor validation

    chunks: TextChunks = execute_llm_chunking(
        model="gpt-4o",
        response_model=TextChunks,
        messages=[
            {"role": "system", "content": "You are an AI assistant tasked with chunking a text into cohesive sections. Your goal is to create chunks that maintain topic coherence and context. Write the context in Turkish."},
            {"role": "user", "content": f"Here's a text with numbered artifacts. Determine the best chunks by specifying start and end artifact numbers. Make the chunks as large as possible while maintaining coherence. Provide a thorough context for each chunk, including information from the rest of the text to ensure the chunk makes good sense. Here's the text:\n\n{text_with_artifacts}"}
        ]
    )

    # Create the final chunked entries with enhanced information
    chunked_entries = []
    for i, chunk in enumerate(chunks.chunks, start=1):
        # Join the sentences in the chunk into a single string
        chunk_text = " ".join(sentences[chunk.start:chunk.end+1])
        enhanced_chunk = EnhancedChunk(
            order=i,
            start=chunk.start,
            end=chunk.end,
            text=chunk_text,
            context=chunk.context
        )
        chunked_entries.append(enhanced_chunk)

    return chunked_entries