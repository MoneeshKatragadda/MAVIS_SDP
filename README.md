**Download the AI Model**
Due to space contraints we were unable to push the model into the repository.
Before implementing the project:
1. Extract the contents of MAVIS folder downloaded from github.
2. Create a folder named "models" inside the MAVIS_SDP-main
3. Download the ai model from the following link: https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q4_K_M.gguf.
4. Add the downloaded model into the models folder.

**Step by step execution (Modeule-wise)**
1. Paste the input story inside input/story.txt
2. Run main.py to obtain the events.json and characters.json
3. Run generate_audio.py to generate:
           a. Master Reference audio for each characeter
           b. Narration or dialogue audio from the story
           c. Background music for each scene
           d. Sound Effects
4. Run generate_cast.py to obtain Canonical images of each character.
5. Run generate_images.py to generate story images.
6. Run movie.py to assemble everything together and generate the final cinematic output

**Step by step execution (Integrated)**
1. Paste the input story inside input/story.txt
2. Run run_pipeline.py (This will run all the steps mentioned in Module-wise execution automatically)
