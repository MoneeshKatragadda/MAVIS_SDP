import os
import shutil
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path

_original_load = torch.load

def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _patched_load

from styletts2 import tts

model = tts.StyleTTS2()
to_mel = torchaudio.transforms.MelSpectrogram(n_mels=80, n_fft=2048, win_length=1200, hop_length=300)

def custom_compute_style(path):
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    wave_tensor = torch.from_numpy(audio).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) + 4) / 4
    
    if hasattr(model, 'device'):
        mel_tensor = mel_tensor.to(model.device)
        
    with torch.no_grad():
        if hasattr(model, 'model') and hasattr(model.model, 'style_encoder'):
            ref_s = model.model.style_encoder(mel_tensor.unsqueeze(0))
            ref_p = model.model.predictor_encoder(mel_tensor.unsqueeze(0))
        elif type(model) is dict and 'style_encoder' in model:
            ref_s = model['style_encoder'](mel_tensor.unsqueeze(0))
            ref_p = model['predictor_encoder'](mel_tensor.unsqueeze(0))
        else:
            raise RuntimeError("Model structure not recognized.")
            
    return torch.cat([ref_s, ref_p], dim=1)

model.compute_style = custom_compute_style

def extract_features(audio_dir, model):
    audio_dir = Path(audio_dir)
    file_paths = []
    style_vectors = []
    
    for wav_file in audio_dir.rglob("*.wav"):
        file_paths.append(str(wav_file))
        with torch.no_grad():
            style_vector = model.compute_style(str(wav_file))
        style_vectors.append(style_vector)
        
    return file_paths, style_vectors

def cluster_and_sort_audio(file_paths, style_vectors, output_base_dir, num_clusters=5):
    if not file_paths or not style_vectors:
        return

    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    vectors_for_kmeans = []
    for vec in style_vectors:
        vec_np = vec.cpu().detach().numpy()
        vectors_for_kmeans.append(vec_np.flatten())
    
    vectors_np = np.array(vectors_for_kmeans)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors_np)
    
    # Optional: Generate a 2D scatter plot using PCA
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors_np)
    
    plt.figure(figsize=(10, 8))
    for i in range(num_clusters):
        cluster_points = vectors_2d[labels == i]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {i}')
        
    plt.title(f'K-Means Clustering of Style Vectors ({output_base_dir.name})')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    plot_path = output_base_dir/'clustering_plot.png'
    plt.savefig(plot_path)
    plt.close()
    
    for path_str, label in zip(file_paths, labels):
        cluster_dir = output_base_dir / f"cluster_{label}"
        cluster_dir.mkdir(parents=True, exist_ok=True)
        
        file_name = Path(path_str).name
        dest_path = cluster_dir / file_name
        shutil.copy2(path_str, dest_path)

def calculate_steering_vector(neutral_dir, emotional_dir, model, output_pt_path):
    neutral_dir = Path(neutral_dir)
    emotional_dir = Path(emotional_dir)
    
    def get_average_vector(directory):
        vectors = []
        for wav_file in directory.rglob("*.wav"):
            with torch.no_grad():
                style_vector = model.compute_style(str(wav_file))
                vectors.append(style_vector)
        
        if not vectors:
            raise ValueError(f"No .wav files found in {directory}")
            
        stacked_vectors = torch.stack(vectors)
        return torch.mean(stacked_vectors, dim=0)

    mean_neutral = get_average_vector(neutral_dir)
    mean_emotional = get_average_vector(emotional_dir)
    steering_vector = mean_emotional - mean_neutral
    
    torch.save(steering_vector, output_pt_path)
    return steering_vector

def run_clustering_pipeline(model, source_audio_dir, clustered_output_dir):
    file_paths, style_vectors = extract_features(source_audio_dir, model)
    cluster_and_sort_audio(file_paths, style_vectors, clustered_output_dir, num_clusters=5)

def process_all_datasets(model):
    datasets = [
        (Path("Audio_Files/male"), Path("Clustered_Audio/male")),
        (Path("Audio_Files/female"), Path("Clustered_Audio/female"))
    ]
    
    for source_dir, output_dir in datasets:
        if source_dir.exists():
            run_clustering_pipeline(model, source_dir, output_dir)

if __name__ == "__main__":
    process_all_datasets(model)