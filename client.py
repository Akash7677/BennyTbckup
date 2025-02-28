import requests
import base64
import numpy as np
import sounddevice as sd
from typing import Optional, Dict, List, Any, Generator
import json
import threading
import queue
import time
from pydub import AudioSegment
from io import BytesIO


class AudioStreamClient:
    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.audio_queue = queue.Queue()
        self.is_playing = False
        self.text_chunks = []
        self.word_timings = []
        self.complete_text = ""
        self._response_complete = threading.Event()
        self._audio_complete = threading.Event()

    def _convert_mp3_to_pcm(self, mp3_data: bytes) -> tuple:
        """Convert MP3 data to PCM format"""
        try:
            # Load MP3 data
            audio = AudioSegment.from_mp3(BytesIO(mp3_data))

            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())

            # Convert to float32 between -1 and 1
            samples = samples.astype(np.float32) / (2 ** 15 if audio.sample_width == 2 else 2 ** 7)

            # Convert to stereo if mono
            if audio.channels == 1:
                samples = np.column_stack((samples, samples))

            return samples, audio.frame_rate
        except Exception as e:
            print(f"Error converting MP3 to PCM: {e}")
            return np.zeros((1000, 2), dtype=np.float32), 44100  # Return silent audio on error

    def _play_audio_worker(self):
        """Worker thread to continuously play audio chunks"""
        print("Starting audio playback worker thread")
        try:
            while self.is_playing or not self.audio_queue.empty():
                try:
                    # Try to get a chunk from the queue
                    try:
                        audio_chunk = self.audio_queue.get(timeout=0.5)
                        if audio_chunk is not None:
                            print(f"Playing audio chunk ({len(audio_chunk)} bytes)")

                            # Convert MP3 to PCM
                            samples, frame_rate = self._convert_mp3_to_pcm(audio_chunk)

                            # Play the audio and wait for it to complete
                            sd.play(samples, frame_rate)
                            sd.wait()  # This blocks until playback is complete
                            print("Audio chunk playback finished")
                    except queue.Empty:
                        # If queue is empty but response isn't complete, keep waiting
                        if not self._response_complete.is_set():
                            time.sleep(0.1)
                            continue

                        # If response is complete and queue is empty, signal audio complete
                        if self._response_complete.is_set() and self.audio_queue.empty():
                            print("Audio playback complete - no more chunks")
                            self._audio_complete.set()
                            break

                        # If queue is empty and we're no longer playing, exit
                        if not self.is_playing:
                            break

                except Exception as e:
                    print(f"Error in audio playback loop: {e}")

            # Make sure we set audio complete event when exiting the thread
            if not self._audio_complete.is_set():
                print("Audio worker thread exiting, setting audio complete event")
                self._audio_complete.set()

        except Exception as e:
            print(f"Critical error in audio worker thread: {e}")
            self._audio_complete.set()  # Ensure we don't deadlock

    def _process_response_chunk(self, chunk_data: Dict[str, Any]) -> None:
        """Process a single response chunk from the server"""
        # Add text chunk to our list of chunks
        if "text_response" in chunk_data:
            # Only add text that isn't already in our complete text to avoid duplication
            new_text = chunk_data["text_response"]
            if new_text not in self.complete_text:
                self.text_chunks.append(new_text)
                self.complete_text = "".join(self.text_chunks)
                print(f"Received text chunk: '{new_text[:30]}...'")

        # Store word timing data
        if "words" in chunk_data and "word_start_times_seconds" in chunk_data and "word_end_times_seconds" in chunk_data:
            # Add word timings that aren't already in our list
            existing_words = set(item["word"] for item in self.word_timings)
            for word, start, end in zip(
                    chunk_data["words"],
                    chunk_data["word_start_times_seconds"],
                    chunk_data["word_end_times_seconds"]
            ):
                timing_entry = {"word": word, "start": start, "end": end}
                if word not in existing_words:
                    self.word_timings.append(timing_entry)

        # Queue audio data for playback
        if "audio_base64" in chunk_data:
            audio_data = base64.b64decode(chunk_data["audio_base64"])
            print(f"Queueing audio chunk ({len(audio_data)} bytes)")
            self.audio_queue.put(audio_data)

        # Check if this is the final chunk
        if chunk_data.get("is_final", False):
            print("Received final chunk signal")
            self._response_complete.set()

    def _process_streaming_response(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """Process streaming response data from the server"""
        buffer = b""

        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue

            buffer += chunk

            # Split by newlines to handle multiple JSON objects
            parts = buffer.split(b'\n')

            # Process all complete parts except the last one (which might be incomplete)
            for part in parts[:-1]:
                if not part.strip():
                    continue

                try:
                    chunk_data = json.loads(part)
                    self._process_response_chunk(chunk_data)
                    yield chunk_data
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}, data: {part}")
                except Exception as e:
                    print(f"Error processing response chunk: {e}")

            # Keep the last part in the buffer (might be incomplete)
            buffer = parts[-1]

        # Process any remaining data in the buffer
        if buffer.strip():
            try:
                chunk_data = json.loads(buffer)
                self._process_response_chunk(chunk_data)
                yield chunk_data
            except json.JSONDecodeError:
                print(f"Error decoding final JSON: {buffer}")
            except Exception as e:
                print(f"Error processing final response chunk: {e}")

    def generate_response(self,
                          text: str,
                          profanity_level: int = 3,
                          voice_id: Optional[str] = "BWIOno0Pi6w2fh1RzS7h",
                          stability: float = 1,
                          similarity_boost: float = 0.5,
                          style: float = 0.7,
                          use_speaker_boost: bool = True,
                          model_provider: str = "OpenAI",
                          wait_for_completion: bool = True) -> Dict:
        """
        Send request to server and handle the streaming response
        """
        # Reset state
        self.text_chunks = []
        self.word_timings = []
        self.complete_text = ""
        self._response_complete.clear()
        self._audio_complete.clear()

        request_data = {
            "text": text,
            "model_provider": "OpenAI",
            "profanity_level": profanity_level,
            "voice_id": voice_id,
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
            "use_speaker_boost": use_speaker_boost
        }

        try:
            # Start audio playback thread first
            self.is_playing = True
            play_thread = threading.Thread(target=self._play_audio_worker)
            play_thread.daemon = True  # Make thread exit when main thread exits
            play_thread.start()

            # Make request to server
            response = requests.post(
                f"{self.server_url}/generate",
                json=request_data,
                stream=True,
                timeout=(3.05, None)  # Connect timeout of 3.05s, no read timeout for streaming
            )
            response.raise_for_status()
            print(response.text)

            # Process the streaming response in a separate thread
            response_thread = threading.Thread(target=lambda: list(self._process_streaming_response(response)))
            response_thread.daemon = True
            response_thread.start()

            # Wait for completion if requested
            if wait_for_completion:
                print("Waiting for response to complete...")
                self._response_complete.wait()
                print("Response complete, waiting for audio to finish...")

                # Now wait for audio to complete
                self._audio_complete.wait(timeout=30)  # 30-second max wait to prevent hanging
                print("Audio playback complete")

            # Return current state information
            return {
                'text_response': self.complete_text,
                'words': [item["word"] for item in self.word_timings],
                'word_start_times_seconds': [item["start"] for item in self.word_timings],
                'word_end_times_seconds': [item["end"] for item in self.word_timings],
                'is_complete': self._response_complete.is_set()
            }

        except Exception as e:
            print(f"Error during request: {e}")
            self.stop_playback()
            return None
        finally:
            # Make sure we don't leave is_playing set to True when we exit
            if wait_for_completion:
                self.is_playing = False

    def stop_playback(self):
        """Stop the audio playback"""
        self.is_playing = False
        sd.stop()
        with self.audio_queue.mutex:
            self.audio_queue.queue.clear()
        self._audio_complete.set()


# Example usage
if __name__ == "__main__":
    client = AudioStreamClient()

    # Test the client with streaming
    response = client.generate_response(
        text="Tell me a really bad joke. you fuckin moron",
        profanity_level=1,
        stability=0.5,
        similarity_boost=0.5,
    )

    print("\nFinal response:", client.complete_text)
    print("\nWord Timings (first 5):", client.word_timings[:5])