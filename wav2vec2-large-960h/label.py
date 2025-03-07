from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import torch, torchaudio
import json
import chromadb

with open("../config.json", mode = "r") as f:
    data = json.load(f)
    SAMPLING_RATE = data["sampling_rate"]
    SEGMENT_LEN = data["segment_length"]
    OVERLAP_LEN = data["overlap_length"]
    DB_PATH = data["database_path"]
    COLLECTION_NAME = data["collection_name"]

fineTunedExtractor = AutoFeatureExtractor.from_pretrained("checkpoints-15-5/checkpoint-32094")
fineTunedModel = AutoModelForAudioClassification.from_pretrained("checkpoints-15-5/checkpoint-32094", device_map = "cuda")

chroma_client = chromadb.PersistentClient(path = DB_PATH)
collection = chroma_client.get_collection(name = COLLECTION_NAME)

waveform, sample_rate = torchaudio.load("../test_audio.mp3")
resampler = torchaudio.transforms.Resample(orig_freq = sample_rate, new_freq = SAMPLING_RATE)
waveform = resampler(waveform)[0]

num_chunks = len(waveform) // (SAMPLING_RATE * SEGMENT_LEN)

for i in range(num_chunks):
    start_sample = i * (SAMPLING_RATE * SEGMENT_LEN)
    end_sample = (i + 1) * (SAMPLING_RATE * SEGMENT_LEN)
    segment = waveform[start_sample : end_sample]
    segment = fineTunedExtractor(segment, sampling_rate = SAMPLING_RATE)["input_values"][0]
    inputs = torch.tensor(segment).unsqueeze(0).to(torch.device("cuda"))
    with torch.no_grad():
        outputs = fineTunedModel(inputs, output_hidden_states = True)
        hidden_states = outputs.hidden_states

    embeddings = hidden_states[-1]
    embeddings = embeddings.mean(dim = 1)[0].cpu().numpy()
    
    res = collection.query(
        query_embeddings = [embeddings],
        n_results = 3
    )
    
    print(f"======= {(i * 10) + 1}-{(i + 1) * 10} Seconds =======")
    for r in res["metadatas"][0]:
        for k, v in r.items():
            print(f"{k}: {v}")
        print("----------------------------")
    print("======================================================")

    del inputs
    torch.cuda.empty_cache()