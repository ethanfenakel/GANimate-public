import os
import numpy as np
import insightface
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from tqdm import tqdm
import argparse

# Initialize ArcFace model
model = insightface.app.FaceAnalysis(allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=0, det_thresh=0.01, det_size=(256, 256))  # Removed 'nms' argument


def extract_embeddings_from_static_images(people_names):

    embeddings = {}
    for person in people_names:
        img = cv2.imread(f'./PTI/smiling_front/{person}.jpeg')
        if img is None:
            print(f"Warning: Could not read image {person}")
            continue

        # Detect faces and get embeddings
        faces = model.get(img)
        if faces:
            embedding = faces[0].embedding
            embeddings[person] = embedding
        else:
            print(f"No face detected in {person}")

    return embeddings

def extract_embeddings_for_cross_id(directory_path, ref_embs):
    # all_embeddings = {}
    all_csims = []
    video_dirs = os.listdir(directory_path)
    for video_dir in tqdm(video_dirs):
        curr_video_people_dirs = os.listdir(os.path.join(directory_path, video_dir))
        curr_video_people_dirs = [f for f in curr_video_people_dirs if f.startswith('0')]
        curr_vid_csims = []
        for person in tqdm(curr_video_people_dirs):
            ref_emb = ref_embs[person]
            curr_video_person_dir = os.path.join(directory_path, video_dir,person)
            image_files = sorted(
            [f for f in os.listdir(curr_video_person_dir) if f.endswith(('png', 'jpg', 'jpeg')) and not f.endswith('_lm.jpg')])

            i = 0
            # Now `embeddings` contains the embeddings for all detected faces
            for image_file in tqdm(image_files):
                    if i > 405:
                        break
                    i += 1
                    image_path = os.path.join(curr_video_person_dir, image_file)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Warning: Could not read image {image_file}")
                        continue

                    # Detect faces and get embeddings
                    faces = model.get(img)
                    if faces:
                        embedding = faces[0].embedding
                        csim = cosine_similarity([embedding], [ref_emb])[0][0]
                        all_csims.append(csim)
                        curr_vid_csims.append(csim)
                    else:
                        print(f"No face detected in {image_file}")
            np.save(os.path.join(directory_path, video_dir,'csims.npy'), np.array(curr_vid_csims))

    return all_csims


# Function to extract embeddings from a directory
def extract_embeddings_from_directory(directory_path):
    embeddings = []
    image_files = sorted(
        [f for f in os.listdir(directory_path) if f.endswith(('png', 'jpg', 'jpeg')) and not f.endswith('_lm.jpg')])

    i = 0
    # Now `embeddings` contains the embeddings for all detected faces
    for image_file in tqdm(image_files):
        if i > 405:
            break
        i+=1
        image_path = os.path.join(directory_path, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_file}")
            continue

        # Detect faces and get embeddings
        faces = model.get(img)
        if faces:
            embedding = faces[0].embedding
            embeddings.append(embedding)
        else:
            print(f"No face detected in {image_file}")

    return np.array(embeddings)

def calculate_cosine_similarity_cross_id(ref, gen):
    all_sims = []
    for person, ref_emb in ref.items():
        gen_embs = gen[person]
        ref_embs = [ref_emb] * len(gen_embs)

        similarities = []
        for emb1, emb2 in zip(gen_embs, ref_embs):
            sim = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append(sim)
        all_sims.append(np.mean(similarities))
    return np.mean(np.array(all_sims))

# Function to calculate cosine similarity between two sets of embeddings
def calculate_cosine_similarity(embeddings1, embeddings2, video_dir):
    if len(embeddings1) != len(embeddings2):
        print("Number of embeddings does not match.")
        # return None
        min_len = min(len(embeddings1), len(embeddings2))
        embeddings1 = embeddings1[:min_len]
        embeddings2 = embeddings2[:min_len]
        print('Not equal lengths')

    similarities = []
    for emb1, emb2 in zip(embeddings1, embeddings2):
        sim = cosine_similarity([emb1], [emb2])[0][0]
        similarities.append(sim)

    return np.mean(similarities)


# Function to compute cosine similarity for a dataset of videos
def calculate_cosine_similarity_for_dataset(generated_videos_dir, reference_videos_dir):
    video_dirs = os.listdir(generated_videos_dir)
    results = {}

    for video_dir in video_dirs:
        gen_video_path = os.path.join(generated_videos_dir, video_dir)
        ref_video_path = os.path.join(reference_videos_dir, video_dir+'.mp4')

        if not os.path.exists(ref_video_path):
            print(f"Reference directory {ref_video_path} not found!")
            continue

        # Extract embeddings for each video
        gen_embeddings = extract_embeddings_from_directory(gen_video_path)
        ref_embeddings = extract_embeddings_from_directory(ref_video_path)

        # Calculate cosine similarity
        cosine_similarity_score = calculate_cosine_similarity(ref_embeddings, gen_embeddings, video_dir)
        if cosine_similarity_score is not None:
            results[video_dir] = cosine_similarity_score
            print(f"Cosine Similarity for {video_dir}: {cosine_similarity_score}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run AED and APD")
    parser.add_argument("--orig_path", type=str, required=True, help="orig_path_of_videos")
    parser.add_argument("--gen_path", type=str, required=True, help="gen_path_of_videos")

    args = parser.parse_args()
    ref_embeddings = extract_embeddings_from_static_images(args.orig_path)
    all_csims = extract_embeddings_for_cross_id(args.gen_path, ref_embeddings)

    csim = np.mean(np.array(all_csims))
    print(f'csim is {csim}')

