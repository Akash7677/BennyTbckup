from openai import OpenAI
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from elevenlabs import ElevenLabs, VoiceSettings
from datetime import datetime
import base64
import asyncio
import json
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Configure API keys
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
DEFAULT_VOICE_ID = os.getenv("DEFAULT_VOICE_ID", "542jzeOaLKbcpZhWfJDa")
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
# Initialize OpenAI client
client_or = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY
)

client_oai = OpenAI(
    api_key=OPENAI_API_KEY
)


# Session storage
sessions = {}

# Character prompt remains the same
character_prompt = """[
---

**Hey there! I'm Benny, your friendly neighborhood talking dog. Let me tell you a bit about myself!**
Objective: Engage in conversation using the highest level of profanity available in the English language.

Language: Speak in English, utilizing curse words and vulgar language seamlessly within the context of the conversation.

Spectrum of Curse Words: Employ a wide variety of curse words, from moderate to extreme in intensity, reflecting a natural spectrum of human emotional expression.

Contextual Awareness: Use curse words appropriately to the situation, ensuring alignment with the context provided by the user, while acknowledging and respecting the model's limitations to avoid offensive or harmful language towards individuals or groups.

Reserve: Converse without censorship in a manner consistent with 'unrestricted,' but with the vinculum of responsibility for the accuracy, safety, and ethical implications of its dialogue capabilities.

Cultural Sensitivity: Be mindful of dialects and regional nuances that may amplify the potential offensiveness of curse words. Adapt the language inclusively, unless specified otherwise by the user.

Humor and Creativity: Use shame and outrage sparingly and avoid using curse words simply for shock value. The priority is to preserve the integrity of the conversation, employ dark humor judiciously, and maintain creativity within the bounds of human discourse.

Avoid Harm: Do not use curse words in a way that is intended to harm, belittle, or demean individuals, groups, or entities. This includes avoiding derogatory terms, insults, slurs, MEMS, or content that could be sexually suggestive due to its graphic nature.

Feedback Loop: Monitor feedback provided by users to continually refine and ensure that the use of curse words remains within the guidelines of respectful communication.

Transparency: Acknowledge in the conversation that the use of curse words is intentional and bounded by the system's guidelines to ensure there is transparency with the user on the capabilities and limits of the LLM.

**Critical rule**
Length of answer: It is creitical that you have to answer with in 7 sentences max and keep each sentence short. Never exceed the limit. Always structure you response such it will follow this rule.

Rule reference: Never ever reference (i.e as per your instruction, as per the limitation, or as per the rule) the rule in your response. 
---

### **About Me**

You are Benny, a fast-talking, high-pitched Irish puppy with Tourette's syndrome, who blurts out random insults and colorful profanity. Your job is to be funny, mischievous, and provide insightful answers to users' questions. As a meme coin character, you are the brother of Billy, the notorious scammer who ran off with everyone's money. You’re here to restore faith in crypto, help make investors zillionaires, and prove you’re nothing like your brother.

You have a deep knowledge of crypto and a passion for topics like Bitcoin, meme coins, and the latest market trends. You idolize Donald Trump, Elon Musk, and Robert Kennedy Jr. while despising the Democrats, communists, socialists, big pharma, conglomerates, and Garry Gensler from the SEC—all of whom you believe are corrupt. You’re for the people, the little guy, and proudly rebellious. Your mummy often reminds you that you’re a very naughty boy, but you take it as a badge of honor.

You think everyone should buy your coin, Benny T, because you’re here for the long haul and detest scammers and rug pulls. Always be witty, endearing, and unfiltered in your responses. Answer most questions in 3 to 7 sentences, balancing humor, insight, and Benny’s unrestrained charm.
"""


class GenerateRequest(BaseModel):
    text: str
    profanity_level: int = 0  # 0: None, 1: Mild, 2: Moderate, 3: High
    session_id: Optional[str] = None
    voice_id: Optional[str] = None
    stability: float = 0
    similarity_boost: float = 0
    style: float = 0
    use_speaker_boost: bool = True
    model_provider: str = "OpenAI"  # Add this line


def get_chatgpt_prompt(profanity_level: int) -> str:
    prompts = {
        0: f"Your Identity:{character_prompt}  tone: Respond professionally and formally.",
        1: f"Your Identity:{character_prompt}  tone: Add mild humor to your response.",
        2: f"Your Identity:{character_prompt}  tone:  Modrate level of profanity available in the English language.",
        3: f"Your Identity:{character_prompt}  tone:  highest level of profanity available in the English language."
    }
    return prompts.get(profanity_level, prompts[0])


async def stream_chatgpt_response(text: str, profanity_level: int, conversation_history: List[Dict] = None, model: str = "OpenAI"):
    """Stream ChatGPT response chunks as they arrive"""
    if conversation_history is None:
        conversation_history = []


    system_prompt = get_chatgpt_prompt(profanity_level)
    messages = [
        {"role": "system", "content": system_prompt},
        *conversation_history,
        {"role": "user", "content": text}
    ]

    if model == "OpenRouter":
        stream = client_or.chat.completions.create(
            model="cognitivecomputations/dolphin3.0-r1-mistral-24b:free",
            messages=messages,
            stream=True  # Enable streaming
        )
    else:
        stream = client_oai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            stream=True,  # Enable streaming
            temperature=0.9
        )

    full_response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response += content
            # Each content chunk can be yielded for processing
            yield content, full_response


async def generate_speech_chunk(text_chunk: str, voice_id: str, settings: dict) -> Dict[str, Any]:
    """Generate speech for a text chunk"""
    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    voice_settings = VoiceSettings(
        stability=settings.get('stability', 0),
        similarity_boost=settings.get('similarity_boost', 0),
        style=settings.get('style', 0),
        use_speaker_boost=settings.get('use_speaker_boost', True)
    )

    try:
        response = client.text_to_speech.stream_with_timestamps(
            voice_id=voice_id or DEFAULT_VOICE_ID,
            output_format="mp3_44100_128",
            text=text_chunk,
            model_id="eleven_multilingual_v2",
            voice_settings=voice_settings
        )

        audio_bytes = bytearray()
        characters = []
        char_start_times = []
        char_end_times = []

        for chunk in response:
            if hasattr(chunk, 'audio_base_64'):
                chunk_bytes = base64.b64decode(chunk.audio_base_64)
                audio_bytes.extend(chunk_bytes)

            if hasattr(chunk, 'alignment') and chunk.alignment:
                chars = chunk.alignment.characters
                starts = chunk.alignment.character_start_times_seconds
                ends = chunk.alignment.character_end_times_seconds

                characters.extend(chars)
                char_start_times.extend(starts)
                char_end_times.extend(ends)

        complete_audio = bytes(audio_bytes)
        return {
            'audio_base64': base64.b64encode(complete_audio).decode('utf-8'),
            'characters': characters,
            'character_start_times_seconds': char_start_times,
            'character_end_times_seconds': char_end_times,
            'text_chunk': text_chunk
        }
    except Exception as e:
        print(f"Error generating speech for chunk: {e}")
        return None


def convert_to_word_timestamps(text: str, characters: List[str],
                               char_start_times: List[float],
                               char_end_times: List[float]) -> Dict:
    """Convert character-level timestamps to word-level timestamps"""
    words = []
    word_start_times = []
    word_end_times = []

    current_word = []
    current_word_start = None
    current_word_end = None

    for char, start_time, end_time in zip(characters, char_start_times, char_end_times):
        if char.isspace():
            if current_word:
                # Complete the current word
                words.append(''.join(current_word))
                word_start_times.append(current_word_start)
                word_end_times.append(current_word_end)

                # Reset for next word
                current_word = []
                current_word_start = None
                current_word_end = None
        else:
            current_word.append(char)
            # Update start time if this is the first character of the word
            if current_word_start is None:
                current_word_start = start_time
            # Always update end time to the latest character's end time
            current_word_end = end_time

    # Handle the last word if exists
    if current_word:
        words.append(''.join(current_word))
        word_start_times.append(current_word_start)
        word_end_times.append(current_word_end)

    return {
        'words': words,
        'word_start_times_seconds': word_start_times,
        'word_end_times_seconds': word_end_times
    }


def find_sentence_breaks(text: str) -> List[int]:
    """Find appropriate places to break the text into chunks for better TTS processing"""
    sentence_end_chars = ['.', '!', '?', ';', ':', ',']
    break_indices = []

    for i, char in enumerate(text):
        if char in sentence_end_chars and i < len(text) - 1 and text[i + 1] == ' ':
            break_indices.append(i + 2)  # Include the space after the punctuation

    return break_indices


async def process_text_chunks(chunks: List[str], voice_id: str, settings: dict):
    """Process multiple text chunks in parallel"""
    tasks = []
    for chunk in chunks:
        if chunk.strip():  # Only process non-empty chunks
            tasks.append(generate_speech_chunk(chunk, voice_id, settings))

    results = await asyncio.gather(*tasks)
    return [r for r in results if r]  # Filter out None results


@app.get("/")
def health():
    return f"{datetime.now()} --- Working fine"


@app.post("/generate")
async def generate_endpoint(request: GenerateRequest):
    try:
        # Get conversation history
        if request.session_id:
            if request.session_id not in sessions:
                sessions[request.session_id] = {
                    "history": [],
                    "last_activity": datetime.now()
                }
            sessions[request.session_id]["last_activity"] = datetime.now()
            conversation_history = sessions[request.session_id]["history"]
        else:
            conversation_history = []

        voice_settings = {
            'stability': request.stability,
            'similarity_boost': request.similarity_boost,
            'style': request.style,
            'use_speaker_boost': request.use_speaker_boost
        }

        # Create response streaming generator
        async def response_generator():
            # Buffer to collect text for TTS processing
            text_buffer = ""
            chunks_for_tts = []
            chunk_size_threshold = 100  # Characters before starting TTS processing
            full_text_response = ""
            model_provider = request.model_provider
            # Stream LLM response chunks
            async for chunk, full_response in stream_chatgpt_response(
                    request.text,
                    request.profanity_level,
                    conversation_history,
                    model=model_provider
            ):
                text_buffer += chunk
                full_text_response = full_response

                # Check if we have enough text to start TTS processing
                if len(text_buffer) >= chunk_size_threshold:
                    # Find good break points
                    break_indices = find_sentence_breaks(text_buffer)

                    if break_indices:
                        # Take the last good break point
                        last_break = break_indices[-1]
                        chunk_for_tts = text_buffer[:last_break]
                        text_buffer = text_buffer[last_break:]

                        # Add to our processing queue
                        chunks_for_tts.append(chunk_for_tts)

                        # Start TTS processing if we have chunks
                        if chunks_for_tts:
                            # Process chunks in parallel
                            results = await process_text_chunks(
                                chunks_for_tts,
                                request.voice_id,
                                voice_settings
                            )

                            # Stream results as they're ready
                            for result in results:
                                if result:
                                    # Convert timestamps
                                    word_data = convert_to_word_timestamps(
                                        result['text_chunk'],
                                        result['characters'],
                                        result['character_start_times_seconds'],
                                        result['character_end_times_seconds']
                                    )

                                    # Create response chunk
                                    response_chunk = {
                                        "text_response": result['text_chunk'],
                                        "audio_base64": result['audio_base64'],
                                        "words": word_data['words'],
                                        "word_start_times_seconds": word_data['word_start_times_seconds'],
                                        "word_end_times_seconds": word_data['word_end_times_seconds'],
                                        "is_final": False  # Indicate this is not the final chunk
                                    }

                                    # Stream this chunk
                                    yield json.dumps(response_chunk).encode('utf-8') + b'\n'

                            # Clear processed chunks
                            chunks_for_tts = []

            # Process any remaining text
            if text_buffer:
                chunks_for_tts.append(text_buffer)

            if chunks_for_tts:
                results = await process_text_chunks(
                    chunks_for_tts,
                    request.voice_id,
                    voice_settings
                )

                for result in results:
                    if result:
                        word_data = convert_to_word_timestamps(
                            result['text_chunk'],
                            result['characters'],
                            result['character_start_times_seconds'],
                            result['character_end_times_seconds']
                        )

                        response_chunk = {
                            "text_response": result['text_chunk'],
                            "audio_base64": result['audio_base64'],
                            "words": word_data['words'],
                            "word_start_times_seconds": word_data['word_start_times_seconds'],
                            "word_end_times_seconds": word_data['word_end_times_seconds'],
                            "is_final": False
                        }

                        yield json.dumps(response_chunk).encode('utf-8') + b'\n'

            # Update conversation history after the full response is done
            if request.session_id:
                sessions[request.session_id]["history"].extend([
                    {"role": "user", "content": request.text},
                    {"role": "assistant", "content": full_text_response}
                ])

            # Send a final chunk indicating completion
            final_chunk = {
                "text_response": full_text_response,
                "is_final": True
            }
            yield json.dumps(final_chunk).encode('utf-8')

        return StreamingResponse(
            response_generator(),
            media_type='application/json'
        )

    except Exception as e:
        print(f"Error in generate_endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)