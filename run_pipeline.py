import concurrent.futures
import logging
import time
import os
import sys

# Ensure imports work if running from this directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from main import run_director
    from generate_images import generate_images
    from generate_audio import generate_audio
except ImportError:
    # Fallback if running from parent
    from phase1.main import run_director
    from phase1.generate_images import generate_images
    from phase1.generate_audio import generate_audio

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [PIPELINE] %(message)s")
logger = logging.getLogger("MAVIS_PIPELINE")

def assemble_movie():
    logger.info("--- PHASE 3: MOVIE ASSEMBLY ---")
    logger.info("This phase would run FFmpeg to stitch validated assets.")
    # In a real implementation, we would read events.json and run ffmpeg commands
    # to combine 'output/images/*.png' and 'output/audio/*.wav' based on timing.

def run_pipeline():
    start_time = time.time()
    logger.info("========================================")
    logger.info("   MAVIS MULTIMODAL PIPELINE START")
    logger.info("========================================")
    
    events_path = "output/events.json"
    
    # Phase 1: Director Engine (LLM + NLP -> Visual DNA -> Events)
    if os.path.exists(events_path):
        logger.info(f"Events file found at {events_path}. Skipping Director Phase.")
    else:
        logger.info(">>> PHASE 1: DIRECTOR ENGINE")
        try:
            run_director()
        except Exception as e:
            logger.error(f"Director Phase Failed: {e}")
            return

    events_path = "output/events.json"
    if not os.path.exists(events_path):
        logger.error("Events file not found. Aborting.")
        return
    
    # Phase 2: Production Factories (Sequential to save VRAM)
    logger.info(">>> PHASE 2: SEQUENTIAL FACTORIES")
    
    # 1. Music/Audio First (Less VRAM usually, cleans up after)
    try:
        generate_audio(events_path)
    except Exception as e:
        logger.error(f"Audio Factory Failed: {e}")

    # 2. Images Second (High VRAM)
    try:
        generate_images(events_path)
    except Exception as e:
        logger.error(f"Image Factory Failed: {e}")
            
    logger.info(">>> PHASE 2 COMPLETE")
    
    # Phase 3: Assembly
    try:
        from movie import generate_movie
        generate_movie(events_path, "output/movie.mp4")
    except ImportError:
        try:
            from phase1.movie import generate_movie
            generate_movie(events_path, "output/movie.mp4")
        except ImportError:
             logger.error("Could not import generate_movie from movie.py")
    
    elapsed = time.time() - start_time
    logger.info(f"========================================")
    logger.info(f"   PIPELINE FINISHED in {elapsed:.2f}s")
    logger.info(f"========================================")

if __name__ == "__main__":
    run_pipeline()
